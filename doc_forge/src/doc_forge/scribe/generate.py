from __future__ import annotations

import requests
import fcntl
from typing import Any
from .config import ScribeConfig

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

    def _call_model(self, prompt: str, temperature: float = 0.2, stop: list[str] | None = None) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": self.cfg.llm_n_predict,
            "temperature": temperature,
            "stop": stop or ["<|im_end|>", "</s>"],
            "stream": False,
        }
        
        # Simple file lock to ensure single model usage
        with open(self.cfg.model_lock_path, "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                resp = requests.post(self.cfg.completion_url, json=payload, timeout=240)
                resp.raise_for_status()
                return (resp.json().get("content") or "").strip()
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def generate(self, rel_path: str, source_text: str, metadata: dict[str, Any]) -> str:
        # Phase 1: Planning (Chain of Thought)
        plan_prompt = self._build_plan_prompt(rel_path, source_text, metadata)
        plan = self._call_model(plan_prompt, temperature=0.4)
        
        # Phase 2: Drafting
        draft_prompt = self._build_draft_prompt(rel_path, source_text, metadata, plan)
        draft = self._call_model(draft_prompt, temperature=0.2)
        
        try:
            return self._normalize(draft)
        except ValueError:
            # Retry once with stricter instructions
            retry_prompt = draft_prompt + "\n\nCRITICAL: Ensure all required headings are present and exact."
            draft = self._call_model(retry_prompt, temperature=0.1)
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
