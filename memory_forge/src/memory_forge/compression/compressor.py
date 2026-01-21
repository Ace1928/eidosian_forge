"""
Semantic Memory Compression.
"""
from typing import List, Protocol
from ..core.interfaces import MemoryItem

class Summarizer(Protocol):
    def summarize(self, text: str) -> str: ...

class MemoryCompressor:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer

    def compress_batch(self, items: List[MemoryItem]) -> MemoryItem:
        """Combine multiple memories into a single summary."""
        if not items:
            raise ValueError("Cannot compress empty list")
        
        combined_text = "\n".join([f"- {item.content}" for item in items])
        summary_text = self.summarizer.summarize(combined_text)
        
        # Create new consolidated memory
        return MemoryItem(
            content=summary_text,
            type=items[0].type,
            metadata={"source_ids": [i.id for i in items], "is_compressed": True},
            importance=max(i.importance for i in items)
        )

