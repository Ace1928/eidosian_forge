#!/usr/bin/env python3
"""
Moltbook Signal Pipeline - High Integrity.
Provides deduplication, schema enforcement, and noise reduction.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Tuple, Set

from .client import MoltbookPost
from .interest import InterestEngine, ScoreBreakdown

logger = logging.getLogger("SignalPipeline")

class SignalPipeline:
    """Infrastructure for cleaning and scoring raw agent signals."""

    def __init__(self, noise_threshold: float = -20.0):
        self.noise_threshold = noise_threshold
        self.engine = InterestEngine()
        self.seen_hashes: Set[str] = set()

    def _fingerprint(self, post: MoltbookPost) -> str:
        """Deterministic hash of post content for deduplication."""
        payload = f"{post.author}:{post.content[:200]}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def process(self, posts: List[MoltbookPost]) -> List[Tuple[MoltbookPost, ScoreBreakdown]]:
        """Filter and score a batch of posts."""
        results = []
        for post in posts:
            h = self._fingerprint(post)
            if h in self.seen_hashes:
                continue
            
            self.seen_hashes.add(h)
            analysis = self.engine.analyze_post(post)
            
            # Critical noise filter
            if analysis.total < self.noise_threshold:
                logger.debug(f"Dropped noise post: {post.id} (Score: {analysis.total})")
                continue
                
            results.append((post, analysis))

        # Memory management for set
        if len(self.seen_hashes) > 5000:
            self.seen_hashes = set(list(self.seen_hashes)[-2500:])
            
        return results

    def process_batch(self, posts: List[MoltbookPost]) -> List[Tuple[MoltbookPost, ScoreBreakdown]]:
        """Compatibility wrapper for older callers/tests."""
        return self.process(posts)
