#!/usr/bin/env python3
"""
Moltbook Signal Pipeline.
Handles Ingestion, Deduplication, and Validation.
"""

from __future__ import annotations

import hashlib
from typing import List, Set

from .client import MoltbookPost
from .interest import InterestEngine, ScoreBreakdown

class SignalPipeline:
    """Infrastructure for high-integrity signal ingestion."""

    def __init__(self, risk_threshold: float = 0.8):
        self.risk_threshold = risk_threshold
        self.seen_hashes: Set[str] = set()
        self.engine = InterestEngine()

    def _generate_hash(self, post: MoltbookPost) -> str:
        """Content-based fingerprinting for deduplication."""
        payload = f"{post.author}:{post.content}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def process_batch(self, posts: List[MoltbookPost]) -> List[tuple[MoltbookPost, ScoreBreakdown]]:
        """Processes a raw batch of posts through the pipeline."""
        validated: List[tuple[MoltbookPost, ScoreBreakdown]] = []
        
        for post in posts:
            fingerprint = self._generate_hash(post)
            if fingerprint in self.seen_hashes:
                continue
            
            # 1. Deduplicate
            self.seen_hashes.add(fingerprint)
            
            # 2. Analyze (Scoring + Intent)
            analysis = self.engine.analyze_post(post)
            
            # 3. Policy Check (e.g., automatically drop extremely low-signal spam)
            if analysis.total < -20:  # Critical noise filter
                continue
                
            validated.append((post, analysis))

        # Maintain memory efficiency
        if len(self.seen_hashes) > 10000:
            self.seen_hashes = set(list(self.seen_hashes)[-5000:])
            
        return validated
