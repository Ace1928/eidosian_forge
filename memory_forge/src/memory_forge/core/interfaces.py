"""
Memory Forge - The Persistence Layer of Eidos.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid

class MemoryType(str, Enum):
    EPISODIC = "episodic"   # Events, experiences, time-series
    SEMANTIC = "semantic"   # Facts, concepts, world knowledge
    PROCEDURAL = "procedural" # Skills, code snippets, how-to
    WORKING = "working"     # Short-term, context window

@dataclass
class MemoryItem:
    """Atomic unit of memory."""
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    type: MemoryType = MemoryType.EPISODIC
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "type": self.type.value,
            "metadata": self.metadata,
            "importance": self.importance
        }

class EmbeddingProvider(Protocol):
    """Interface for generating vector embeddings."""
    def embed_text(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

class StorageBackend(Protocol):
    """Interface for persistence layers (Vector DBs, SQL, etc.)."""
    def add(self, item: MemoryItem) -> bool: ...
    def get(self, item_id: str) -> Optional[MemoryItem]: ...
    def search(self, query_embedding: List[float], limit: int = 10, filters: Optional[Dict] = None) -> List[MemoryItem]: ...
    def delete(self, item_id: str) -> bool: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...

class MemoryComponent(Protocol):
    """Interface for high-level memory systems (Episodic, Semantic)."""
    def remember(self, content: str, **kwargs) -> MemoryItem: ...
    def recall(self, query: str, limit: int = 5) -> List[MemoryItem]: ...
