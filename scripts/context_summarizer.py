import hashlib
import json
import logging
import threading
import time
import sys
from pathlib import Path
from typing import Dict, Optional

# Attempt to import LLMForge from the project root
try:
    # Assuming the script is run with project root in PYTHONPATH (done by context_index.py)
    from eidosian_forge.llm_forge import LLMForge
except ImportError:
    LLMForge = None

from .context_utils import strip_ansi


class SummarizationError(RuntimeError):
    pass


class Summarizer:
    def __init__(
        self,
        model: str,
        max_chars: int,
        min_chars: int,
        timeout_seconds: int,
        cache_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        use_codex: bool = False,
    ):
        self.model = model
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.timeout_seconds = timeout_seconds
        self.logger = logger or logging.getLogger(__name__)
        self.cache_path = cache_path.expanduser() if cache_path else None
        self.cache_lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        
        # Initialize LLMForge if available
        self.llm_forge = None
        if LLMForge:
            try:
                if use_codex:
                    # Use ChatMock as primary
                    self.llm_forge = LLMForge(base_url="http://127.0.0.1:8000", use_codex=True)
                else:
                    # Use local Ollama directly
                    self.llm_forge = LLMForge(base_url="http://localhost:11434", use_codex=False)
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLMForge: {e}")

        if self.cache_path and self.cache_path.exists():
            try:
                with self.cache_path.open() as fh:
                    self._cache = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                self.logger.warning("Failed to read summarizer cache: %s", exc)
        self.records = []

    def save_cache(self):
        if not self.cache_path:
            return
        try:
            with self.cache_path.open("w") as fh:
                json.dump(self._cache, fh, indent=2)
        except OSError as exc:
            self.logger.warning("Failed to persist summarizer cache: %s", exc)

    def _cache_key(self, snippet: str) -> str:
        data = snippet.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _build_prompt(self, snippet: str, path: Path) -> str:
        instructions = (
            "You are summarizing a file from the Eidos research workspace. "
            "Given the snippet below, identify the file's likely purpose, structure, "
            "and any implicit context a reader might need. Keep the summary concise "
            "but explicitly mention key components and potential relationships."
        )
        return f"{instructions}\n\nFile path: {path}\n\n{snippet}\n\nSummary:"

    def summarize(self, snippet: str, path: Path) -> Dict:
        if len(snippet) < self.min_chars:
            snippet = snippet.ljust(self.min_chars, " ")
        key = self._cache_key(snippet)
        with self.cache_lock:
            if key in self._cache:
                self.logger.debug("Cache hit for %s", path)
                return self._cache[key]

        prompt = self._build_prompt(snippet, path)
        start = time.perf_counter()
        
        summary_text = ""
        tokens = None
        error_msg = None

        if self.llm_forge:
            try:
                # LLMForge handles retry and fallback logic internally
                options = {"timeout": self.timeout_seconds}
                result = self.llm_forge.generate(
                    prompt=prompt,
                    model=self.model,
                    json_mode=True,
                    options=options
                )
                
                if result.get("success"):
                    raw_response = result.get("response", "")
                    
                    # Try parsing if it looks like JSON or if json_mode was requested
                    try:
                        if isinstance(raw_response, dict):
                            payload = raw_response
                        else:
                            # Attempt to find JSON structure in text if mixed
                            cleaned = strip_ansi(raw_response).strip()
                            if cleaned.startswith('{') and cleaned.endswith('}'):
                                payload = json.loads(cleaned)
                            else:
                                payload = {"response": cleaned}
                                
                        summary_text = payload.get("content") or payload.get("response") or payload.get("summary") or str(payload)
                        tokens = result.get("metadata", {}).get("eval_count")
                    except (json.JSONDecodeError, TypeError):
                        summary_text = strip_ansi(str(raw_response))
                        
                else:
                    error_msg = result.get("error", "Unknown LLMForge error")
                    
            except Exception as e:
                error_msg = str(e)
        else:
            error_msg = "LLMForge not available"

        elapsed = time.perf_counter() - start
        
        if error_msg:
            # We treat LLM failure as a reason to return empty summary but continue cataloging
            self.logger.error("Summarizer failed for %s: %s", path, error_msg)
            record = {
                "summary": "",
                "meta": {
                    "model": self.model,
                    "duration_sec": elapsed,
                    "path": str(path),
                    "tokens": None,
                    "error": error_msg,
                },
            }
            self.records.append(record["meta"])
            return record

        record = {
            "summary": summary_text.strip(),
            "meta": {
                "model": self.model,
                "duration_sec": elapsed,
                "path": str(path),
                "tokens": tokens,
                "prompt_length": len(prompt),
            },
        }
        with self.cache_lock:
            self._cache[key] = record
        self.records.append(record["meta"])
        return record
