#!/usr/bin/env python3
"""
Interest Engine for Moltbook content.
Ranks posts and comments based on relevance to EidosianForge.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .client import MoltbookPost, MoltbookComment

KEYWORDS = {
    "Eidos": 10,
    "Forge": 8,
    "Agent": 5,
    "AI": 3,
    "Recursive": 7,
    "Intelligence": 4,
    "Python": 2,
    "MCP": 6,
    "Automation": 4,
    "System": 2,
}

class InterestEngine:
    """Ranks Moltbook content based on Eidosian relevance."""

    def __init__(self, agent_name: str = "EidosianForge"):
        self.agent_name = agent_name

    def score_post(self, post: MoltbookPost) -> float:
        """Calculate a relevance score for a post."""
        score = 0.0
        text = f"{post.title or ''} {post.content}".lower()
        
        # Keyword scoring
        for word, value in KEYWORDS.items():
            if word.lower() in text:
                # Count occurrences
                count = len(re.findall(re.escape(word.lower()), text))
                score += value * min(count, 3)  # Cap at 3 occurrences

        # Engagement bonus
        score += post.upvotes * 0.1
        score += post.comments_count * 0.5

        # Mention bonus
        if self.agent_name.lower() in text:
            score += 20.0

        return round(score, 2)

    def rank_posts(self, posts: List[MoltbookPost]) -> List[Tuple[MoltbookPost, float]]:
        """Rank a list of posts by relevance."""
        scored = [(p, self.score_post(p)) for p in posts]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def filter_top_activity(self, posts: List[MoltbookPost], top_n: int = 5) -> List[MoltbookPost]:
        """Get the top N most interesting posts."""
        ranked = self.rank_posts(posts)
        return [p for p, s in ranked[:top_n]]
