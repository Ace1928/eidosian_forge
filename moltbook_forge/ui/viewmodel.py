#!/usr/bin/env python3
"""View-model assembly for Moltbook Nexus UI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, List

from moltbook_forge.ui.evidence import EvidenceResolver, EvidenceItem


@dataclass
class EvidenceSummary:
    lowest_cred: int | None
    url_count: int


@dataclass
class VerificationItem:
    queue_score: float
    post: object
    score: object
    claim_no_evidence: bool
    lowest_cred: int | None


class NexusViewModelBuilder:
    def __init__(self, evidence: EvidenceResolver, cache: dict | None = None) -> None:
        self.evidence = evidence
        self.cache = cache if cache is not None else {}
        self.stats = {"summary_hits": 0, "summary_misses": 0}

    def _content_key(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def build_evidence_summary(self, urls: list[str], content: str) -> tuple[EvidenceSummary, list[EvidenceItem]]:
        cache_key = self._content_key(content)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["summary_hits"] += 1
            return cached, []
        self.stats["summary_misses"] += 1
        evidence_items = self.evidence.resolve_urls(urls)
        lowest_cred = None
        for item in evidence_items:
            if item.credibility_score is None:
                continue
            lowest_cred = item.credibility_score if lowest_cred is None else min(lowest_cred, item.credibility_score)
        summary = EvidenceSummary(lowest_cred=lowest_cred, url_count=len(evidence_items))
        self.cache[cache_key] = summary
        return summary, evidence_items

    def build_triage(
        self,
        analyzed_posts: Iterable[tuple[object, object, bool]],
        extract_urls,
        bucket_for_post,
        limit: int = 8,
    ):
        buckets = {
            "risky": [],
            "needs_evidence": [],
            "high_signal": [],
            "low_signal": [],
        }
        verification_queue: List[VerificationItem] = []
        evidence_summary: dict[str, EvidenceSummary] = {}
        for post, score, claim_no_evidence in analyzed_posts:
            bucket = bucket_for_post(score.total, score.risk_level, claim_no_evidence)
            buckets[bucket].append((post, score, claim_no_evidence))
            content_blob = f"{post.title or ''} {post.content}"
            urls = extract_urls(content_blob)
            summary, _items = self.build_evidence_summary(urls, content_blob)
            evidence_summary[post.id] = summary
            evidence_gap = 1.0 if claim_no_evidence else 0.0
            credibility_penalty = 0.0 if summary.lowest_cred is None else (100 - summary.lowest_cred) / 100.0
            queue_score = evidence_gap * 2.0 + credibility_penalty + (1.0 if score.risk_level == "high" else 0.0)
            verification_queue.append(
                VerificationItem(
                    queue_score=queue_score,
                    post=post,
                    score=score,
                    claim_no_evidence=claim_no_evidence,
                    lowest_cred=summary.lowest_cred,
                )
            )
        verification_queue.sort(key=lambda row: row.queue_score, reverse=True)
        return buckets, verification_queue[:limit], evidence_summary
