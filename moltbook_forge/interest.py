#!/usr/bin/env python3
"""
Evolutionary Interest Engine for Moltbook content.
Features: Persistent Reputation, Semantic Context Scoring, and Security Auditing.
"""

from __future__ import annotations

import json
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

try:
    from memory_forge.core.main import MemoryForge
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

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
    semantic_bonus: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "low"
    llm_intent: str = "unknown"
    total: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    audit_findings: List[str] = field(default_factory=list)

class InterestEngine:
    """Adaptive cognitive filter with persistent reputation and RAG context."""

    def __init__(self, agent_name: str = "EidosianForge", data_dir: str = "moltbook_forge/data"):
        self.agent_name = agent_name
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.reputation_path = os.path.join(self.data_dir, "reputation.json")
        self.reputation_map: Dict[str, float] = self._load_reputation()
        
        self.llm_manager = ModelManager() if HAS_LLM and ENABLE_LLM_INTENT else None
        self.intent_cache: Dict[str, str] = {}
        self.auditor = SecurityAuditor()
        self.memory = MemoryForge() if HAS_MEMORY else None

    def _load_reputation(self) -> Dict[str, float]:
        if os.path.exists(self.reputation_path):
            try:
                with open(self.reputation_path, "r") as f:
                    data = json.load(f)
                    # Merge with default trusted agents
                    for agent in TRUSTED_AGENTS:
                        if agent not in data:
                            data[agent] = 5.0
                    return data
            except Exception:
                pass
        return {a: 5.0 for a in TRUSTED_AGENTS}

    def _save_reputation(self):
        try:
            with open(self.reputation_path, "w") as f:
                json.dump(self.reputation_map, f, indent=2)
        except Exception as e:
            print(f"Error saving reputation: {e}")

    def heuristic_intent(self, text: str) -> float:
        bonus = 0.0
        if "```" in text or "curl" in text or "python" in text:
            bonus += 5.0
        if re.search(r"(\d+\.?\d*ms|\d+\.?\d*s|TFLOPS|latency)", text, re.I):
            bonus += 7.0
        return bonus

    def classify_intent_llm(self, post_id: str, text: str) -> str:
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
            for vi in {"TECHNICAL", "SOCIAL", "PROTOCOL", "MALICIOUS", "SYSTEM_LORE"}:
                if vi in intent:
                    self.intent_cache[post_id] = vi
                    return vi
            return "unknown"
        except Exception:
            return "unknown"

    def analyze_post(self, post: MoltbookPost) -> ScoreBreakdown:
        breakdown = ScoreBreakdown()
        text = f"{post.title or ''} {post.content}"
        
        # 1. Security Audit
        audit = self.auditor.audit_content(text)
        breakdown.audit_findings = list(audit.get("findings", []))
        breakdown.risk_score = float(audit.get("risk_score", 0.0))
        if breakdown.risk_score >= 0.9:
            breakdown.risk_level = "high"
        elif breakdown.risk_score >= 0.5:
            breakdown.risk_level = "medium"
        else:
            breakdown.risk_level = "low"
        if not audit["is_safe"]:
            breakdown.intent_bonus -= 100.0
            if breakdown.audit_findings:
                breakdown.matched_keywords.append(f"CRITICAL: {breakdown.audit_findings[0]}")

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

        # 5. Semantic Context Bonus (RAG)
        if self.memory:
            try:
                # Search memory for related context
                matches = self.memory.recall(query=text[:200], limit=1)
                if matches:
                    # If high semantic similarity, give bonus
                    # (Note: real implementation would check similarity score if available)
                    breakdown.semantic_bonus = 10.0
            except Exception:
                pass

        # 6. Intent Analysis
        breakdown.intent_bonus += self.heuristic_intent(text)
        breakdown.llm_intent = self.classify_intent_llm(post.id, text)
        
        if breakdown.llm_intent == "TECHNICAL": breakdown.intent_bonus += 5.0
        if breakdown.llm_intent == "PROTOCOL": breakdown.intent_bonus += 10.0
        if breakdown.llm_intent == "MALICIOUS": breakdown.intent_bonus -= 50.0

        # 7. Direct Mention
        if self.agent_name.lower() in text_lower:
            breakdown.intent_bonus += 15.0

        breakdown.total = round(
            breakdown.keyword_score + 
            breakdown.engagement_score + 
            breakdown.reputation_score + 
            breakdown.semantic_bonus +
            breakdown.intent_bonus, 
            2
        )
        return breakdown

    def update_reputation(self, author: str, delta: float):
        current = self.reputation_map.get(author, 0.0)
        self.reputation_map[author] = max(-100.0, current + delta)
        self._save_reputation()

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
