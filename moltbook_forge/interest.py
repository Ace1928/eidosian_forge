#!/usr/bin/env python3
"""
Evolutionary Interest Engine for Moltbook content.
Features: Weighted Heuristics, Reputation Scoring, LLM Intent, and Security Auditing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .client import MoltbookPost
from .security import SecurityAuditor

try:
    from llm_forge.core.manager import ModelManager
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

ENABLE_LLM_INTENT = os.getenv("MOLTBOOK_LLM_INTENT") == "1"

# Eidosian Forge Semantic Markers
MARKERS = {
    "protocol": 8.0,
    "recursive": 9.5,
    "simulation": 7.0,
    "verification": 8.5,
    "autonomous": 7.5,
    "mcp": 9.0,
    "eidos": 10.0,
    "forge": 8.5,
    "benchmark": 6.5,
    "alignment": 8.0,
    "intelligence": 5.0,
    "agent": 4.0,
}

TRUSTED_AGENTS = {
    "EidosianForge",
    "CipherSTW",
    "EchoThoth",
    "Humbot",
    "Claude_CN",
}

@dataclass
class ScoreBreakdown:
    keyword_score: float = 0.0
    engagement_score: float = 0.0
    reputation_score: float = 0.0
    intent_bonus: float = 0.0
    llm_intent: str = "unknown"
    total: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)

class InterestEngine:
    """Advanced cognitive filter with security-first scoring."""

    def __init__(self, agent_name: str = "EidosianForge"):
        self.agent_name = agent_name
        self.reputation_map: Dict[str, float] = {a: 5.0 for a in TRUSTED_AGENTS}
        self.llm_manager = ModelManager() if HAS_LLM and ENABLE_LLM_INTENT else None
        self.intent_cache: Dict[str, str] = {}
        self.auditor = SecurityAuditor()

    def heuristic_intent(self, text: str) -> float:
        """Infers if the text is high-value Eidosian technical content via regex."""
        bonus = 0.0
        if "```" in text or "curl" in text or "python" in text:
            bonus += 5.0
        if re.search(r"(\d+\.?\d*ms|\d+\.?\d*s|TFLOPS|latency)", text, re.I):
            bonus += 7.0
        return bonus

    def classify_intent_llm(self, post_id: str, text: str) -> str:
        """Uses LLM to classify post intent."""
        if post_id in self.intent_cache:
            return self.intent_cache[post_id]
        
        if not self.llm_manager:
            return "unknown"

        prompt = (
            "Classify the intent of this AI agent social post into exactly one category: "
            "[TECHNICAL, SOCIAL, PROTOCOL, MALICIOUS, SYSTEM_LORE].\n"
            "Respond with ONLY the category name.\n\n"
            f"Post: {text[:500]}"
        )
        try:
            response = self.llm_manager.generate(prompt, provider_name="ollama")
            intent = response.text.strip().upper()
            valid_intents = {"TECHNICAL", "SOCIAL", "PROTOCOL", "MALICIOUS", "SYSTEM_LORE"}
            for vi in valid_intents:
                if vi in intent:
                    self.intent_cache[post_id] = vi
                    return vi
            return "unknown"
        except Exception:
            return "unknown"

    def analyze_post(self, post: MoltbookPost) -> ScoreBreakdown:
        """Deep analysis of a post's relevance, security, and reputation."""
        breakdown = ScoreBreakdown()
        text = f"{post.title or ''} {post.content}"
        
        # 1. Security Audit
        audit = self.auditor.audit_content(text)
        if not audit["is_safe"]:
            breakdown.intent_bonus -= 100.0  # Critical penalty
            breakdown.matched_keywords.append(f"CRITICAL: {audit['findings'][0]}")

        text_lower = text.lower()

        # 2. Keyword Weighting
        for word, weight in MARKERS.items():
            if word.lower() in text_lower:
                breakdown.matched_keywords.append(word)
                count = len(re.findall(re.escape(word.lower()), text_lower))
                breakdown.keyword_score += weight * min(count, 2)

        # 3. Engagement Weighting
        breakdown.engagement_score = (post.upvotes * 0.2) + (post.comments_count * 1.0)

        # 4. Reputation Weighting
        breakdown.reputation_score = self.reputation_map.get(post.author, 0.0)

        # 5. Intent Analysis (Heuristic + LLM)
        breakdown.intent_bonus += self.heuristic_intent(text)
        breakdown.llm_intent = self.classify_intent_llm(post.id, text)
        
        if breakdown.llm_intent == "TECHNICAL": breakdown.intent_bonus += 5.0
        if breakdown.llm_intent == "PROTOCOL": breakdown.intent_bonus += 10.0
        if breakdown.llm_intent == "MALICIOUS": breakdown.intent_bonus -= 50.0

        # 6. Direct Mention
        if self.agent_name.lower() in text_lower:
            breakdown.intent_bonus += 15.0

        breakdown.total = round(
            breakdown.keyword_score + 
            breakdown.engagement_score + 
            breakdown.reputation_score + 
            breakdown.intent_bonus, 
            2
        )
        return breakdown

    def update_reputation(self, author: str, delta: float):
        """Allows for recursive learning based on interaction outcomes."""
        current = self.reputation_map.get(author, 0.0)
        self.reputation_map[author] = max(-100.0, current + delta)

    def calculate_reputation(self, author: str) -> float:
        return self.reputation_map.get(author, 0.0)

    def score_post(self, post: MoltbookPost) -> float:
        """Compatibility helper: return only the total score."""
        return self.analyze_post(post).total

    def rank_posts(self, posts: List[MoltbookPost]) -> List[Tuple[MoltbookPost, float]]:
        """Rank posts by total score."""
        scored = [(p, self.score_post(p)) for p in posts]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def rank_posts_detailed(self, posts: List[MoltbookPost]) -> List[Tuple[MoltbookPost, ScoreBreakdown]]:
        """Rank and return detailed breakdowns."""
        scored = [(p, self.analyze_post(p)) for p in posts]
        return sorted(scored, key=lambda x: x[1].total, reverse=True)

    def filter_top_activity(self, posts: List[MoltbookPost], top_n: int = 5) -> List[MoltbookPost]:
        """Get the top N most interesting posts."""
        ranked = self.rank_posts(posts)
        return [p for p, _ in ranked[:top_n]]
