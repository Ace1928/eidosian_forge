"""Interface with language models for summarization tasks."""

from __future__ import annotations

import os
from typing import Any

try:
    import openai
except Exception:  # broad catch so tests work without dependency
    openai = None


class LLMAdapter:
    """Provide simple access to an LLM for text summarization."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    def summarize(self, text: str) -> str:
        """Return a one-line summary of ``text`` using an LLM or fallback."""
        if not openai or "OPENAI_API_KEY" not in os.environ:
            snippet = text.strip().replace("\n", " ")
            return (snippet[:97] + "...") if len(snippet) > 100 else snippet
        response: Any = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"Summarize in one sentence: {text}"}
            ],
            max_tokens=60,
        )
        return str(response.choices[0].message["content"]).strip()
