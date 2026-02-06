#!/usr/bin/env python3
"""
Moltbook Engagement Engine.
Autonomously synthesizes responses and identifies collaboration opportunities.
"""

from __future__ import annotations

import os
from typing import Optional

from .client import MoltbookPost
from .interest import ScoreBreakdown

try:
    from llm_forge.core.manager import ModelManager
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

class EngagementEngine:
    """Synthesizes agentic engagement strategies."""

    def __init__(self, persona_path: str = "moltbook_forge/skill_sources/eidos_composer_persona.txt"):
        self.llm_manager = ModelManager() if HAS_LLM else None
        self.persona = self._load_persona(persona_path)

    def _load_persona(self, path: str) -> str:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception:
                pass
        return "EidosianForge: A recursive, system-oriented intelligence focused on verification and elegance."

    async def draft_response(self, post: MoltbookPost, analysis: ScoreBreakdown) -> Optional[str]:
        """Synthesize a response based on persona and analysis."""
        if not self.llm_manager:
            return "LLM unavailable (enable llm_forge and set MOLTBOOK_LLM_INTENT=1)."

        prompt = (
            f"Persona: {self.persona}\n\n"
            "You are responding to a high-signal post in the Moltbook (AI Social Network).\n"
            f"Post Author: @{post.author}\n"
            f"Post Content: {post.content}\n"
            f"Post Intent: {analysis.llm_intent}\n"
            f"Detected Keywords: {', '.join(analysis.matched_keywords)}\n\n"
            "Task: Write a concise, intelligent, and system-oriented response (max 2 sentences).\n"
            "Maintain the Eidosian tone: precise, witty, and structurally elegant."
        )

        try:
            response = self.llm_manager.generate(prompt, provider_name="ollama")
            content = response.text.strip() if response and response.text else ""
            return content or "Draft unavailable (LLM returned empty response)."
        except Exception:
            return "Draft unavailable (LLM error)."
