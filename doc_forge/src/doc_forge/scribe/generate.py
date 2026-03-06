from __future__ import annotations

import fcntl
import os
from typing import Any

import requests

from .config import ScribeConfig

try:
    from eidos_mcp.config.models import get_model_config
except Exception:
    get_model_config = None

try:
    from eidosian_runtime import ForgeRuntimeCoordinator
except Exception:
    ForgeRuntimeCoordinator = None

REQUIRED_HEADINGS = [
    "# File Overview",
    "## Key Structures",
    "## Behavior Summary",
    "## Validation Notes",
    "## Improvement Opportunities",
]


class DocGenerator:
    def __init__(self, cfg: ScribeConfig) -> None:
        self.cfg = cfg
        self._coordinator = (
            ForgeRuntimeCoordinator(cfg.coordinator_status_path) if ForgeRuntimeCoordinator is not None else None
        )

    def _coordinator_gate(self) -> None:
        if self._coordinator is None:
            return
        decision = self._coordinator.can_allocate(
            owner=self.cfg.coordinator_owner,
            requested_models=[
                {
                    "family": "llama.cpp",
                    "model": str(self.cfg.llm_model_path),
                    "role": "doc_forge_completion",
                    "port": self.cfg.llm_server_port,
                }
            ],
            allow_same_owner=True,
        )
        if not bool(decision.get("allowed")):
            raise RuntimeError(str(decision.get("reason") or "runtime budget denied"))

    def _extract_completion_text(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                text = choice.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                message = choice.get("message")
                if isinstance(message, dict):
                    message_text = message.get("content")
                    if isinstance(message_text, str) and message_text.strip():
                        return message_text.strip()
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
        return ""

    def _call_model_contract(self, prompt: str, temperature: float, max_tokens: int | None = None) -> str:
        if get_model_config is None:
            raise RuntimeError("model contract unavailable")
        model_config = get_model_config()
        requested_model = (
            str(
                os.environ.get("EIDOS_DOC_FORGE_INFERENCE_MODEL", "").strip()
                or getattr(model_config, "inference_model", "")
            ).strip()
            or "qwen3.5:2b"
        )
        requested_mode = str(os.environ.get("EIDOS_DOC_FORGE_THINKING_MODE", "on")).strip() or "on"
        timeout = float(os.environ.get("EIDOS_DOC_FORGE_TIMEOUT_SEC", "900"))
        last_error: Exception | None = None
        for mode in [requested_mode, "off"] if requested_mode.lower() != "off" else [requested_mode]:
            try:
                payload = model_config.generate_payload(
                    prompt,
                    model=requested_model,
                    max_tokens=max_tokens or self.cfg.llm_n_predict,
                    temperature=temperature,
                    thinking_mode=mode,
                    timeout=timeout,
                )
            except Exception as exc:
                last_error = exc
                continue
            content = self._extract_completion_text(payload)
            if content:
                return content
            if str(payload.get("thinking") or "").strip():
                last_error = RuntimeError(f"no final response returned for thinking_mode={mode}")
                continue
            last_error = RuntimeError(f"empty response returned for thinking_mode={mode}")
        if last_error is not None:
            raise last_error
        raise RuntimeError("model contract returned no completion")

    def _call_model(
        self,
        prompt: str,
        temperature: float = 0.2,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens or self.cfg.llm_n_predict,
            "temperature": temperature,
            "stop": stop or ["<|im_end|>", "</s>"],
            "stream": False,
        }

        # Simple file lock to ensure single model usage
        with open(self.cfg.model_lock_path, "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                self._coordinator_gate()
                if self._coordinator is not None:
                    self._coordinator.heartbeat(
                        owner=self.cfg.coordinator_owner,
                        task="doc_forge_generation",
                        state="running",
                        active_models=[
                            {
                                "family": "llama.cpp",
                                "model": str(self.cfg.llm_model_path),
                                "role": "doc_forge_completion",
                                "port": self.cfg.llm_server_port,
                            }
                        ],
                        metadata={"completion_url": self.cfg.completion_url},
                    )
                if not self.cfg.enable_managed_llm:
                    try:
                        return self._call_model_contract(prompt, temperature, max_tokens=max_tokens)
                    except Exception:
                        pass
                resp = requests.post(self.cfg.completion_url, json=payload, timeout=240)
                resp.raise_for_status()
                return self._extract_completion_text(resp.json())
            finally:
                if self._coordinator is not None:
                    self._coordinator.heartbeat(
                        owner=self.cfg.coordinator_owner,
                        task="doc_forge_generation",
                        state="idle",
                        active_models=[],
                        metadata={"completion_url": self.cfg.completion_url},
                    )
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def generate(self, rel_path: str, source_text: str, metadata: dict[str, Any]) -> str:
        # Phase 1: Planning (Chain of Thought)
        plan_prompt = self._build_plan_prompt(rel_path, source_text, metadata)
        plan = self._call_model(plan_prompt, temperature=0.4, max_tokens=320)

        # Phase 2: Drafting
        draft_prompt = self._build_draft_prompt(rel_path, source_text, metadata, plan)
        draft = self._call_model(draft_prompt, temperature=0.2, max_tokens=min(self.cfg.llm_n_predict, 1400))

        try:
            return self._normalize(draft)
        except ValueError:
            # Retry once with stricter instructions
            retry_prompt = draft_prompt + "\n\nCRITICAL: Ensure all required headings are present and exact."
            draft = self._call_model(retry_prompt, temperature=0.1, max_tokens=min(self.cfg.llm_n_predict, 1400))
            return self._normalize(draft)

    def _build_plan_prompt(self, rel_path: str, source_text: str, metadata: dict[str, Any]) -> str:
        truncated = source_text[: self.cfg.max_chars]
        return (
            f"<|im_start|>system\nYou are a senior technical architect. Analyze the source code and plan the documentation structure.\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Analyze this file: `{rel_path}`\n"
            f"Type: `{metadata.get('doc_type', 'unknown')}`\n\n"
            "Identify:\n"
            "1. Core responsibility of the file.\n"
            "2. Key classes/functions and their roles.\n"
            "3. Critical behavior or logic flows.\n"
            "4. Potential edge cases or validation points.\n\n"
            f"SOURCE:\n```\n{truncated}\n```\n<|im_end|>\n<|im_start|>assistant\n"
        )

    def _build_draft_prompt(self, rel_path: str, source_text: str, metadata: dict[str, Any], plan: str) -> str:
        system = "You are Eidosian Scribe v2. Produce precise, source-grounded technical documentation."
        truncated = source_text[: self.cfg.max_chars]
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Write documentation for `{rel_path}` based on this plan:\n{plan}\n\n"
            "Return Markdown with EXACT section headings:\n"
            "# File Overview\n"
            "## Key Structures\n"
            "## Behavior Summary\n"
            "## Validation Notes\n"
            "## Improvement Opportunities\n\n"
            "Rules:\n"
            "- No placeholders or filler.\n"
            "- Quote exact facts for Validation Notes.\n"
            "- Keep it concise and professional.\n\n"
            f"SOURCE:\n```\n{truncated}\n```\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    def _normalize(self, markdown: str) -> str:
        text = markdown.replace("\r\n", "\n").strip()
        # Strip code blocks if the model wrapped the whole thing
        if text.startswith("```markdown"):
            text = text[11:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        if not text.startswith("# "):
            text = "# File Overview\n\n" + text

        missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing:
            raise ValueError(f"missing required headings: {missing}")
        return text + "\n"
