"""
Tiered Memory System for EIDOS.

This module implements a multi-tiered memory architecture:
- SHORT_TERM: Recent context, session-specific, auto-expires
- WORKING: Task-relevant memories actively being processed
- LONG_TERM: Persistent episodic and semantic memories
- SELF: EIDOS identity, lessons, introspection
- USER: User profiles, preferences, interaction patterns

The system enables automatic memory promotion/demotion based on
relevance, importance, and access patterns.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

from eidosian_core import eidosian
from eidosian_core.ports import get_service_url

from .interfaces import MemoryType


class MemoryTier(str, Enum):
    """Memory tiers with different persistence characteristics."""

    SHORT_TERM = "short_term"  # Session-specific, volatile
    WORKING = "working"  # Task-relevant, currently active
    LONG_TERM = "long_term"  # Persistent episodic/semantic
    SELF = "self"  # EIDOS identity and lessons
    USER = "user"  # User profiles and preferences


class MemoryNamespace(str, Enum):
    """Namespaces for organizing memories by context."""

    EIDOS = "eidos"  # EIDOS self-knowledge
    USER = "user"  # User-specific memories
    TASK = "task"  # Task/session memories
    KNOWLEDGE = "knowledge"  # General knowledge
    CODE = "code"  # Code-related memories
    CONVERSATION = "conversation"  # Dialogue memories


@dataclass
class TieredMemoryItem:
    """Extended memory item with tiering support."""

    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tier: MemoryTier = MemoryTier.SHORT_TERM
    namespace: MemoryNamespace = MemoryNamespace.TASK
    memory_type: MemoryType = MemoryType.EPISODIC
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    ttl_seconds: Optional[int] = None  # Time-to-live for expiration
    tags: Set[str] = field(default_factory=set)
    linked_memories: Set[str] = field(default_factory=set)

    @eidosian()
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @eidosian()
    def is_expired(self) -> bool:
        """Check if memory has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    @eidosian()
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "tier": self.tier.value,
            "namespace": self.namespace.value,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "importance": self.importance,
            "ttl_seconds": self.ttl_seconds,
            "tags": list(self.tags),
            "linked_memories": list(self.linked_memories),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredMemoryItem":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            tier=MemoryTier(data.get("tier", "short_term")),
            namespace=MemoryNamespace(data.get("namespace", "task")),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 1.0),
            ttl_seconds=data.get("ttl_seconds"),
            tags=set(data.get("tags", [])),
            linked_memories=set(data.get("linked_memories", [])),
        )


class EmbeddingService(Protocol):
    """Protocol for embedding generation."""

    def embed_text(self, text: str) -> List[float]: ...


class OllamaEmbedder:
    """
    Ollama-based embedding service using unified model configuration.

    Uses nomic-embed-text by default (768 dimensions, 8192 context).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = get_service_url("ollama_http", default_port=11434, default_host="localhost", default_path=""),
    ):
        # Try to get model from unified config
        try:
            from eidos_mcp.config.models import model_config

            self.model = model or model_config.embedding.model
        except ImportError:
            self.model = model or "nomic-embed-text"
        self.base_url = base_url

    @eidosian()
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        import httpx

        url = f"{self.base_url}/api/embeddings"
        data = {"model": self.model, "prompt": text}

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=data)
                resp.raise_for_status()
                return resp.json()["embedding"]
        except Exception:
            # Return empty embedding on failure (graceful degradation)
            return []

    @eidosian()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(t) for t in texts]


class TieredMemorySystem:
    """
    Multi-tiered memory system for EIDOS.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                 MEMORY ARCHITECTURE                  │
    ├─────────────────────────────────────────────────────┤
    │  SHORT-TERM (Session)    → Auto-expires, volatile   │
    │  WORKING (Active)        → Task-relevant, promoted  │
    │  LONG-TERM (Persistent)  → Compressed, important    │
    │  SELF (Eidos)           → Identity, lessons, growth │
    │  USER (Lloyd)           → Preferences, patterns     │
    └─────────────────────────────────────────────────────┘
    """

    # Default TTL for different tiers (in seconds)
    TIER_TTL = {
        MemoryTier.SHORT_TERM: 3600,  # 1 hour
        MemoryTier.WORKING: 86400,  # 24 hours
        MemoryTier.LONG_TERM: None,  # Permanent
        MemoryTier.SELF: None,  # Permanent
        MemoryTier.USER: None,  # Permanent
    }

    # Importance thresholds for tier promotion
    PROMOTION_THRESHOLD = {
        MemoryTier.SHORT_TERM: 0.6,  # Promote to WORKING
        MemoryTier.WORKING: 0.8,  # Promote to LONG_TERM
    }

    def __init__(
        self,
        persistence_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingService] = None,
    ):
        if persistence_dir is None:
            self.persistence_dir = Path("./data/tiered_memory")
        elif isinstance(persistence_dir, str):
            self.persistence_dir = Path(persistence_dir)
        else:
            self.persistence_dir = persistence_dir
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder

        # Initialize tier storage
        self.tiers: Dict[MemoryTier, Dict[str, TieredMemoryItem]] = {tier: {} for tier in MemoryTier}

        # Load persisted memories
        self._load()

    @eidosian()
    def remember(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        namespace: MemoryNamespace = MemoryNamespace.TASK,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 1.0,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new memory with tiering support."""
        embedding = None
        if self.embedder:
            try:
                embedding = self.embedder.embed_text(content)
            except Exception:
                pass

        item = TieredMemoryItem(
            content=content,
            tier=tier,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            ttl_seconds=self.TIER_TTL.get(tier),
            tags=tags or set(),
            metadata=metadata or {},
        )

        self.tiers[tier][item.id] = item
        self._persist_tier(tier)
        return item.id

    @eidosian()
    def recall(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[List[MemoryTier]] = None,
        namespaces: Optional[List[MemoryNamespace]] = None,
        min_importance: float = 0.0,
    ) -> List[TieredMemoryItem]:
        """Retrieve memories across tiers with semantic search."""
        tiers = tiers or list(MemoryTier)

        # Collect all candidate memories
        candidates: List[TieredMemoryItem] = []
        for tier in tiers:
            for item in self.tiers[tier].values():
                if item.is_expired():
                    continue
                if item.importance < min_importance:
                    continue
                if namespaces and item.namespace not in namespaces:
                    continue
                candidates.append(item)

        if not candidates:
            return []

        # If we have an embedder, do semantic search
        if self.embedder:
            query_vec = self.embedder.embed_text(query)
            scored = []
            for item in candidates:
                if item.embedding:
                    score = self._cosine_similarity(query_vec, item.embedding)
                else:
                    # Fallback to keyword match
                    score = self._keyword_score(query, item)
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [item for _, item in scored[:limit]]
        else:
            # Keyword search fallback with scoring
            scored = []
            for item in candidates:
                score = self._keyword_score(query, item)
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Return items with non-zero score
            results = [item for score, item in scored[:limit] if score > 0]

        # Update access counts
        for item in results:
            item.touch()
            self._check_promotion(item)

        return results

    @eidosian()
    def recall_self(self, query: str, limit: int = 5) -> List[TieredMemoryItem]:
        """Retrieve EIDOS self-memories."""
        return self.recall(
            query,
            limit=limit,
            tiers=[MemoryTier.SELF],
            namespaces=[MemoryNamespace.EIDOS],
        )

    @eidosian()
    def recall_user(self, query: str, user_id: str = "lloyd", limit: int = 5) -> List[TieredMemoryItem]:
        """Retrieve user-specific memories."""
        results = self.recall(
            query,
            limit=limit * 2,  # Get more, filter by user
            tiers=[MemoryTier.USER],
            namespaces=[MemoryNamespace.USER],
        )
        # Filter by user_id in metadata
        return [r for r in results if r.metadata.get("user_id") == user_id][:limit]

    @eidosian()
    def remember_self(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an EIDOS self-memory (identity, lesson, insight)."""
        return self.remember(
            content=content,
            tier=MemoryTier.SELF,
            namespace=MemoryNamespace.EIDOS,
            memory_type=memory_type,
            importance=1.5,  # Self memories are high importance
            tags=tags or {"self", "eidos"},
            metadata=metadata or {},
        )

    @eidosian()
    def remember_user(
        self,
        content: str,
        user_id: str = "lloyd",
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a user-specific memory."""
        meta = metadata or {}
        meta["user_id"] = user_id
        return self.remember(
            content=content,
            tier=MemoryTier.USER,
            namespace=MemoryNamespace.USER,
            memory_type=MemoryType.SEMANTIC,
            importance=1.2,
            tags=tags or {"user", user_id},
            metadata=meta,
        )

    @eidosian()
    def remember_lesson(
        self,
        lesson: str,
        context: str = "",
        outcome: str = "",
        tags: Optional[Set[str]] = None,
    ) -> str:
        """Store a learned lesson for EIDOS self-improvement."""
        content = f"LESSON: {lesson}"
        if context:
            content += f"\nCONTEXT: {context}"
        if outcome:
            content += f"\nOUTCOME: {outcome}"

        all_tags = {"lesson", "learning", "self-improvement"}
        if tags:
            all_tags.update(tags)

        return self.remember_self(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            tags=all_tags,
            metadata={"lesson": lesson, "context": context, "outcome": outcome},
        )

    @eidosian()
    def promote(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Manually promote a memory to a higher tier."""
        for tier in MemoryTier:
            if memory_id in self.tiers[tier]:
                item = self.tiers[tier].pop(memory_id)
                item.tier = target_tier
                item.ttl_seconds = self.TIER_TTL.get(target_tier)
                self.tiers[target_tier][memory_id] = item
                self._persist_tier(tier)
                self._persist_tier(target_tier)
                return True
        return False

    @eidosian()
    def demote(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Demote a memory to a lower tier."""
        return self.promote(memory_id, target_tier)

    @eidosian()
    def link_memories(self, memory_id_a: str, memory_id_b: str) -> bool:
        """Create bidirectional links between memories."""
        item_a = self._find_memory(memory_id_a)
        item_b = self._find_memory(memory_id_b)

        if item_a and item_b:
            item_a.linked_memories.add(memory_id_b)
            item_b.linked_memories.add(memory_id_a)
            self._persist_tier(item_a.tier)
            self._persist_tier(item_b.tier)
            return True
        return False

    @eidosian()
    def get_related(self, memory_id: str) -> List[TieredMemoryItem]:
        """Get memories linked to the specified memory."""
        item = self._find_memory(memory_id)
        if not item:
            return []

        related = []
        for linked_id in item.linked_memories:
            linked_item = self._find_memory(linked_id)
            if linked_item:
                related.append(linked_item)
        return related

    @eidosian()
    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count of removed items."""
        removed = 0
        for tier in MemoryTier:
            expired_ids = [mid for mid, item in self.tiers[tier].items() if item.is_expired()]
            for mid in expired_ids:
                del self.tiers[tier][mid]
                removed += 1
            if expired_ids:
                self._persist_tier(tier)
        return removed

    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        stats = {
            "total": 0,
            "by_tier": {},
            "by_namespace": {},
            "by_type": {},
        }

        for tier in MemoryTier:
            count = len(self.tiers[tier])
            stats["by_tier"][tier.value] = count
            stats["total"] += count

        # Count by namespace and type
        for tier in MemoryTier:
            for item in self.tiers[tier].values():
                ns = item.namespace.value
                mt = item.memory_type.value
                stats["by_namespace"][ns] = stats["by_namespace"].get(ns, 0) + 1
                stats["by_type"][mt] = stats["by_type"].get(mt, 0) + 1

        return stats

    @eidosian()
    def list_all(
        self,
        tiers: Optional[List[MemoryTier]] = None,
        namespaces: Optional[List[MemoryNamespace]] = None,
    ) -> List[TieredMemoryItem]:
        """List all memories, optionally filtered by tier and namespace."""
        tiers = tiers or list(MemoryTier)
        results = []
        for tier in tiers:
            for item in self.tiers[tier].values():
                if namespaces and item.namespace not in namespaces:
                    continue
                if not item.is_expired():
                    results.append(item)
        return results

    def _check_promotion(self, item: TieredMemoryItem) -> None:
        """Check if memory should be promoted based on access patterns."""
        if item.tier in self.PROMOTION_THRESHOLD:
            threshold = self.PROMOTION_THRESHOLD[item.tier]
            # Calculate promotion score based on access count and importance
            score = (item.access_count * 0.1) + item.importance
            if score >= threshold:
                # Determine next tier
                if item.tier == MemoryTier.SHORT_TERM:
                    self.promote(item.id, MemoryTier.WORKING)
                elif item.tier == MemoryTier.WORKING:
                    self.promote(item.id, MemoryTier.LONG_TERM)

    def _find_memory(self, memory_id: str) -> Optional[TieredMemoryItem]:
        """Find a memory across all tiers."""
        for tier in MemoryTier:
            if memory_id in self.tiers[tier]:
                return self.tiers[tier][memory_id]
        return None

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _keyword_score(self, query: str, item: TieredMemoryItem) -> float:
        """Calculate keyword match score for an item."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        score = 0.0
        content_lower = item.content.lower()

        # Exact phrase match
        if query_lower in content_lower:
            score += 2.0

        # Word overlap
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        score += overlap * 0.5

        # Tag match
        for tag in item.tags:
            if query_lower in tag.lower():
                score += 1.0

        # Metadata match (for key fields)
        for key, val in item.metadata.items():
            if isinstance(val, str) and query_lower in val.lower():
                score += 0.5

        return score

    def _persist_tier(self, tier: MemoryTier) -> None:
        """Save a specific tier to disk."""
        path = self.persistence_dir / f"{tier.value}.json"
        data = [item.to_dict() for item in self.tiers[tier].values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load all tiers from disk."""
        for tier in MemoryTier:
            path = self.persistence_dir / f"{tier.value}.json"
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item_data in data:
                        item = TieredMemoryItem.from_dict(item_data)
                        self.tiers[tier][item.id] = item
                except Exception:
                    pass

    @eidosian()
    def save_all(self) -> None:
        """Persist all tiers to disk."""
        for tier in MemoryTier:
            self._persist_tier(tier)
