"""
Central Memory Controller.
"""
from typing import Optional, List, Protocol, Union
from .interfaces import MemoryItem, MemoryType, StorageBackend
try:
    from ..backends.chroma_store import ChromaBackend
    _CHROMA_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    ChromaBackend = None
    _CHROMA_IMPORT_ERROR = exc
from ..backends.json_store import JsonBackend
from .config import MemoryConfig

# Protocol for Embedding Provider
class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]: ...

class MemoryForge:
    def __init__(self, config: Optional[MemoryConfig] = None, embedder: Optional[EmbeddingService] = None):
        self.config = config or MemoryConfig()
        self.embedder = embedder
        
        # Initialize Episodic Backend
        if self.config.episodic.type == "chroma":
            if ChromaBackend is None:
                raise RuntimeError(
                    "Chroma backend requested but chromadb is not installed."
                ) from _CHROMA_IMPORT_ERROR
            self.episodic = ChromaBackend(
                self.config.episodic.collection_name,
                self.config.episodic.connection_string,
            )
        elif self.config.episodic.type == "json":
            self.episodic = JsonBackend(self.config.episodic.connection_string)
        else:
            raise ValueError(f"Unknown backend type: {self.config.episodic.type}")

    def remember(self, content: str, embedding: Optional[List[float]] = None, metadata: dict = None) -> str:
        """Store a new memory. Auto-generates embedding if provider available."""
        if embedding is None:
            if self.embedder:
                embedding = self.embedder.embed_text(content)
            else:
                # If no embedding provided and no embedder, we can't store vector.
                # However, some backends (like JSON) might accept None embedding.
                # Chroma requires it.
                if self.config.episodic.type == "chroma":
                     raise ValueError("No embedding provided and no embedder configured for ChromaDB.")
                embedding = None # JSON backend handles None

        item = MemoryItem(
            content=content, 
            embedding=embedding, 
            metadata=metadata or {},
            type=MemoryType.EPISODIC
        )
        self.episodic.add(item)
        return item.id

    def recall(self, query: Union[str, List[float]], limit: int = 5, filter_metadata: dict = None) -> List[MemoryItem]:
        """Retrieve memories by semantic query (text) or vector."""
        if isinstance(query, str):
            if not self.embedder:
                raise ValueError("Cannot recall by text query without an embedder.")
            query_vec = self.embedder.embed_text(query)
        else:
            query_vec = query

        if query_vec is None and self.config.episodic.type == "json":
             # Fallback for JSON store without embeddings? Not really supported by 'search' interface which expects vec.
             # JSON store search expects list[float].
             return []

        return self.episodic.search(query_vec, limit=limit, filters=filter_metadata)

    def stats(self) -> dict:
        return {
            "episodic_count": self.episodic.count()
        }
