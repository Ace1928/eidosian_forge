import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None  # type: ignore

from .memory import MemoryStore


PRIMARY_MODEL = "Qwen/Qwen3-1.7B-FP8"
FALLBACK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class NarrativeEngine:
    """Self-referential engine backed by a language model."""

    def __init__(
        self,
        memory_path: str = "memory.json",
        think_interval: int = 10,
        model_name: str = PRIMARY_MODEL,
        max_tokens: int = 512,
        adapter_path: str | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.store = MemoryStore(memory_path)
        self.think_interval = think_interval
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.adapter_path = adapter_path
        self.pipeline_factory = pipeline_factory or pipeline
        self._timer: Optional[threading.Timer] = None
        self._pipeline = self._load_model()

    def _load_model(self):
        if not AutoModelForCausalLM:
            raise RuntimeError("transformers package is required to load models")

        model = None
        tokenizer = None
        last_error: Exception | None = None
        for name in [self.model_name, FALLBACK_MODEL]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
                print(f"Loaded local model {name}")
                break
            except Exception as exc_local:
                last_error = exc_local
                print(f"Local load failed for {name}: {exc_local}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForCausalLM.from_pretrained(name)
                    print(f"Loaded remote model {name}")
                    break
                except Exception as exc_remote:
                    last_error = exc_remote
                    print(f"Remote load failed for {name}: {exc_remote}")
                    model = None

        if not model or not tokenizer:
            raise RuntimeError(
                f"Unable to load model {self.model_name}. Last error: {last_error}"
            )

        if self.adapter_path and PeftModel and os.path.exists(self.adapter_path):
            try:
                model = PeftModel.from_pretrained(model, self.adapter_path)
                print(f"Loaded adapter from {self.adapter_path}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load adapter {self.adapter_path}: {exc}") from exc

        return self.pipeline_factory("text-generation", model=model, tokenizer=tokenizer)

    def _reset_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.think_interval, self.free_thought)
        self._timer.daemon = True
        self._timer.start()

    def record_interaction(self, user_input: str, response: str) -> None:
        self.store.data.interactions.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user_input,
                "response": response,
            }
        )
        self._update_glossary(user_input)
        self._reset_timer()
        self.store.save()

    def _update_glossary(self, text: str) -> None:
        for word in text.split():
            word = word.lower().strip(".,!?")
            if not word:
                continue
            self.store.data.glossary[word] = self.store.data.glossary.get(word, 0) + 1

    def respond(self, user_input: str) -> str:
        if not self._pipeline:
            raise RuntimeError("Model pipeline is not available")
        try:
            result = self._pipeline(
                user_input, max_new_tokens=self.max_tokens, do_sample=True
            )
            response = result[0]["generated_text"]
        except Exception as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(f"Model inference failed: {exc}") from exc
        self.record_interaction(user_input, response)
        return response

    def free_thought(self) -> None:
        """Called during idle periods to generate autonomous output."""
        if self.store.data.interactions:
            last = self.store.data.interactions[-1]
            prompt = f"Reflect on this message: {last['user']}"
        else:
            prompt = "Introduce yourself."

        if not self._pipeline:
            raise RuntimeError("Model pipeline is not available")
        try:
            result = self._pipeline(
                prompt, max_new_tokens=self.max_tokens, do_sample=True
            )
            thought = result[0]["generated_text"]
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Model inference failed during free thought: {exc}")
            thought = prompt

        print(f"\n[THOUGHT] {thought}\n> ", end="", flush=True)
        self.store.data.events.append({"timestamp": datetime.utcnow().isoformat(), "event": thought})
        self.store.save()
        self._reset_timer()

    def shutdown(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.store.save()

