"""
Memory Retrieval Engine: generate ranked memory suggestions based on inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from eidosian_core import eidosian

from .interfaces import MemoryItem
from .memory_broker import MemoryBroker


@dataclass
class RetrievalResult:
    source: str
    content: str
    score: float
    metadata: Dict[str, str]


class MemoryRetrievalEngine:
    """Combine episodic recall with broker layers for suggestions."""

    def __init__(self, broker: MemoryBroker):
        self.broker = broker

    @eidosian()
    def suggest(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []

        # Episodic recall
        try:
            episodic: List[MemoryItem] = self.broker.recall_episodic(query, limit=limit)
            for item in episodic:
                results.append(
                    RetrievalResult(
                        source="episodic",
                        content=item.content,
                        score=1.0,
                        metadata=item.metadata if hasattr(item, "metadata") else {},
                    )
                )
        except Exception:
            pass

        # Layered memory (self/user/working/procedural)
        for layer in ["self", "user", "working", "procedural"]:
            for mem in self.broker.recall_layer(layer, limit=limit):
                results.append(
                    RetrievalResult(
                        source=layer,
                        content=mem.content,
                        score=0.5,
                        metadata={k: str(v) for k, v in mem.metadata.items()},
                    )
                )

        # Simple ranking: prefer episodic first, then recency by source
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
