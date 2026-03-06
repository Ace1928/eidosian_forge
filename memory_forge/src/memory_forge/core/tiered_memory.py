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
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

from eidosian_core import eidosian
from eidosian_core.ports import get_service_url
from eidosian_vector import HNSWVectorStore

from .interfaces import MemoryType

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


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
    _WORD_RE = re.compile(r"[A-Za-z0-9_.:-]{3,}")
    _DOMAIN_KEYWORDS = {
        "memory": {"memory", "recall", "remember", "tiered", "episodic", "semantic"},
        "knowledge": {"knowledge", "graph", "graphrag", "community", "ontology"},
        "code": {"code", "function", "class", "module", "snippet", "library"},
        "runtime": {"scheduler", "runtime", "coordinator", "daemon", "service", "port"},
        "model": {"qwen", "ollama", "llama", "embedding", "reasoning", "thinking", "model"},
        "docs": {"docs", "documentation", "markdown", "report", "summary"},
        "consciousness": {"consciousness", "workspace", "attention", "autonomy", "self-model"},
        "termux": {"termux", "android", "prefix", "venv"},
    }

    def __init__(
        self,
        persistence_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingService] = None,
        vector_store_dir: Optional[Path] = None,
        llm_enrichment: bool | None = None,
        llm_model: Optional[str] = None,
        llm_thinking_mode: Optional[str] = None,
    ):
        if persistence_dir is None:
            self.persistence_dir = Path("./data/tiered_memory")
        elif isinstance(persistence_dir, str):
            self.persistence_dir = Path(persistence_dir)
        else:
            self.persistence_dir = persistence_dir
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.llm_enrichment = (
            bool(llm_enrichment)
            if llm_enrichment is not None
            else str(os.environ.get("EIDOS_MEMORY_LLM_ENRICHMENT", "")).strip().lower() in {"1", "true", "on", "yes"}
        )
        self.llm_model = llm_model or str(os.environ.get("EIDOS_MEMORY_LLM_MODEL", "qwen3.5:2b")).strip()
        self.llm_thinking_mode = (
            str(os.environ.get("EIDOS_MEMORY_LLM_THINKING_MODE", llm_thinking_mode or "on")).strip() or "on"
        )
        self._thread_lock = threading.RLock()
        self._lock_path = self.persistence_dir / ".tiered_memory.lock"
        self._lock_handle = None
        self._lock_depth = 0
        self.vector_store = None
        if embedder is not None:
            try:
                store_dir = Path(vector_store_dir) if vector_store_dir else self.persistence_dir / "vectors"
                self.vector_store = HNSWVectorStore(store_dir)
            except Exception:
                self.vector_store = None

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
        with self._mutation_lock():
            self._reload_from_disk_locked()
            return self._remember_locked(
                content=content,
                tier=tier,
                namespace=namespace,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                metadata=metadata,
                embedding=embedding,
            )

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
            if self.vector_store and query_vec:
                filters = {
                    "tier": [tier.value for tier in tiers],
                }
                if namespaces:
                    filters["namespace"] = [ns.value for ns in namespaces]
                vector_hits = self.vector_store.query(query_vec, limit=max(limit * 3, limit), filters=filters)
                seen: Set[str] = set()
                results = []
                for hit in vector_hits:
                    item = self._find_memory(hit.item_id)
                    if item is None or item.id in seen:
                        continue
                    if item.is_expired() or item.importance < min_importance:
                        continue
                    seen.add(item.id)
                    results.append(item)
                    if len(results) >= limit:
                        break
            else:
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
        with self._mutation_lock():
            self._reload_from_disk_locked()
            for tier in MemoryTier:
                if memory_id in self.tiers[tier]:
                    item = self.tiers[tier].pop(memory_id)
                    item.tier = target_tier
                    item.ttl_seconds = self.TIER_TTL.get(target_tier)
                    self.tiers[target_tier][memory_id] = item
                    self._persist_tier(tier)
                    self._persist_tier(target_tier)
                    self._upsert_vector_store(item)
                    return True
            return False

    @eidosian()
    def demote(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Demote a memory to a lower tier."""
        return self.promote(memory_id, target_tier)

    @eidosian()
    def link_memories(self, memory_id_a: str, memory_id_b: str) -> bool:
        """Create bidirectional links between memories."""
        with self._mutation_lock():
            self._reload_from_disk_locked()
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
        with self._mutation_lock():
            self._reload_from_disk_locked()
            removed = 0
            for tier in MemoryTier:
                expired_ids = [mid for mid, item in self.tiers[tier].items() if item.is_expired()]
                for mid in expired_ids:
                    if self.vector_store is not None:
                        self.vector_store.delete(mid)
                    del self.tiers[tier][mid]
                    removed += 1
                if expired_ids:
                    self._persist_tier(tier)
            return removed

    @eidosian()
    def semantic_compress_old_memories(
        self,
        older_than_days: int = 30,
        similarity_threshold: float = 0.55,
        min_cluster_size: int = 3,
        max_clusters: int = 50,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Semantically compress older memories into long-term summary memories.

        The operation is idempotent and non-destructive:
        - source memories are retained
        - compressed sources are tagged with `compressed_into`
        - generated summaries are marked with `is_compressed`
        """
        older_than_days = max(0, int(older_than_days))
        min_cluster_size = max(2, int(min_cluster_size))
        max_clusters = max(1, int(max_clusters))

        with self._mutation_lock():
            self._merge_disk_state_locked()
            return self._semantic_compress_old_memories_locked(
                older_than_days=older_than_days,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                max_clusters=max_clusters,
                dry_run=dry_run,
            )

    def _semantic_compress_old_memories_locked(
        self,
        older_than_days: int,
        similarity_threshold: float,
        min_cluster_size: int,
        max_clusters: int,
        dry_run: bool,
    ) -> Dict[str, Any]:
        """Compression implementation that assumes the persistence lock is held."""

        cutoff = datetime.now() - timedelta(days=older_than_days)
        candidates: List[TieredMemoryItem] = []
        for tier in (MemoryTier.WORKING, MemoryTier.LONG_TERM):
            for item in self.tiers[tier].values():
                if item.created_at > cutoff:
                    continue
                if item.metadata.get("is_compressed"):
                    continue
                if item.metadata.get("compressed_into"):
                    continue
                if item.memory_type not in (MemoryType.EPISODIC, MemoryType.SEMANTIC):
                    continue
                candidates.append(item)

        if len(candidates) < min_cluster_size:
            return {
                "eligible_memories": len(candidates),
                "potential_clusters": 0,
                "clusters_created": 0,
                "summaries_created": 0,
                "source_marked": 0,
                "dry_run": dry_run,
            }

        sorted_candidates = sorted(candidates, key=lambda item: item.created_at)
        pending = {item.id for item in sorted_candidates}
        by_id = {item.id: item for item in sorted_candidates}
        clusters: List[List[TieredMemoryItem]] = []

        for item in sorted_candidates:
            if item.id not in pending:
                continue
            pending.remove(item.id)
            cluster = [item]
            for other in sorted_candidates:
                if other.id not in pending:
                    continue
                if self._memory_similarity(item, other) >= similarity_threshold:
                    cluster.append(other)
                    pending.remove(other.id)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                if len(clusters) >= max_clusters:
                    break
            else:
                # Keep non-clustered memories available as future anchors.
                for member in cluster[1:]:
                    pending.add(member.id)

        if dry_run:
            return {
                "eligible_memories": len(candidates),
                "potential_clusters": len(clusters),
                "clusters_created": 0,
                "summaries_created": 0,
                "source_marked": 0,
                "dry_run": True,
            }

        summaries_created = 0
        source_marked = 0
        touched_tiers: Set[MemoryTier] = set()

        for cluster in clusters:
            source_ids = [member.id for member in cluster]
            summary_content = self._semantic_cluster_summary(cluster)
            summary_tags: Set[str] = {"compressed", "semantic-summary", "auto-compressed"}
            for member in cluster:
                summary_tags.update({tag for tag in member.tags if tag})

            summary_id = self._remember_locked(
                content=summary_content,
                tier=MemoryTier.LONG_TERM,
                namespace=MemoryNamespace.KNOWLEDGE,
                memory_type=MemoryType.SEMANTIC,
                importance=max(member.importance for member in cluster),
                tags=summary_tags,
                metadata={
                    "is_compressed": True,
                    "compression_schema": "semantic_v1",
                    "source_ids": source_ids,
                    "cluster_size": len(cluster),
                    "older_than_days": older_than_days,
                    "similarity_threshold": similarity_threshold,
                    "compressed_at": datetime.now().isoformat(),
                },
                embedding=None,
            )
            summaries_created += 1

            for source_id in source_ids:
                source_item = by_id.get(source_id)
                if source_item is None:
                    source_item = self._find_memory(source_id)
                if source_item is None:
                    continue
                source_item.metadata["compressed_into"] = summary_id
                source_item.metadata["compressed_at"] = datetime.now().isoformat()
                source_item.tags.add("compressed_source")
                touched_tiers.add(source_item.tier)
                source_marked += 1

        for tier in touched_tiers:
            self._persist_tier(tier)

        return {
            "eligible_memories": len(candidates),
            "potential_clusters": len(clusters),
            "clusters_created": len(clusters),
            "summaries_created": summaries_created,
            "source_marked": source_marked,
            "dry_run": False,
        }

    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        stats = {
            "total": 0,
            "by_tier": {},
            "by_namespace": {},
            "by_type": {},
            "vector_count": self.vector_store.count() if self.vector_store is not None else 0,
            "community_count": 0,
            "top_communities": [],
        }
        communities: Dict[str, int] = {}

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
                community = str(item.metadata.get("community") or "")
                if community:
                    communities[community] = communities.get(community, 0) + 1

        stats["community_count"] = len(communities)
        stats["top_communities"] = [
            {"community": name, "count": count}
            for name, count in sorted(communities.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        ]

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

    @eidosian()
    def enrich_memory(self, memory_id: str, use_llm: Optional[bool] = None) -> Dict[str, Any]:
        """Enrich a specific memory with structured tags/metadata/community labels."""
        with self._mutation_lock():
            self._reload_from_disk_locked()
            item = self._find_memory(memory_id)
            if item is None:
                return {"updated": False, "found": False, "memory_id": memory_id}
            updated = self._apply_enrichment(item, use_llm=use_llm)
            self._persist_tier(item.tier)
            self._upsert_vector_store(item)
            return {
                "updated": bool(updated),
                "found": True,
                "memory_id": item.id,
                "community": str(item.metadata.get("community") or ""),
                "keywords": list(item.metadata.get("keywords") or []),
                "tags": sorted(item.tags),
            }

    @eidosian()
    def enrich_all_memories(
        self,
        limit: int = 0,
        tiers: Optional[List[MemoryTier]] = None,
        use_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Batch-enrich memories to improve tagging, grouping, and vector metadata."""
        tiers = tiers or list(MemoryTier)
        max_items = max(0, int(limit))
        updated = 0
        touched: Set[MemoryTier] = set()
        communities: Dict[str, int] = {}
        with self._mutation_lock():
            self._merge_disk_state_locked()
            items: List[TieredMemoryItem] = []
            for tier in tiers:
                items.extend(self.tiers[tier].values())
            items.sort(key=lambda row: row.created_at)
            for item in items:
                if max_items and updated >= max_items:
                    break
                if self._apply_enrichment(item, use_llm=use_llm):
                    updated += 1
                    touched.add(item.tier)
                community = str(item.metadata.get("community") or "")
                if community:
                    communities[community] = communities.get(community, 0) + 1
                self._upsert_vector_store(item)
            for tier in touched:
                self._persist_tier(tier)
        return {
            "updated": updated,
            "scanned": len(items) if "items" in locals() else 0,
            "tiers": [tier.value for tier in tiers],
            "llm_enrichment": bool(self.llm_enrichment if use_llm is None else use_llm),
            "communities": dict(sorted(communities.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        }

    @eidosian()
    def reindex_vector_store(self, limit: int = 0) -> Dict[str, Any]:
        """Rebuild embeddings/vector metadata for all memories."""
        reindexed = 0
        scanned = 0
        max_items = max(0, int(limit))
        with self._mutation_lock():
            self._merge_disk_state_locked()
            for item in self.list_all():
                scanned += 1
                if max_items and reindexed >= max_items:
                    break
                if self.embedder and not item.embedding:
                    try:
                        item.embedding = self.embedder.embed_text(item.content)
                    except Exception:
                        item.embedding = item.embedding or None
                self._apply_enrichment(item, use_llm=False)
                self._upsert_vector_store(item)
                reindexed += 1
            for tier in MemoryTier:
                self._persist_tier(tier)
        return {
            "reindexed": reindexed,
            "scanned": scanned,
            "vector_count": self.vector_store.count() if self.vector_store is not None else 0,
        }

    @eidosian()
    def community_summary(self, limit: int = 20) -> Dict[str, Any]:
        """Summarize memory communities derived from vector-aware enrichment."""
        groups: Dict[str, Dict[str, Any]] = {}
        for item in self.list_all():
            community = str(item.metadata.get("community") or "unclassified").strip() or "unclassified"
            group = groups.setdefault(
                community,
                {
                    "community": community,
                    "count": 0,
                    "tiers": {},
                    "namespaces": {},
                    "top_tags": {},
                    "examples": [],
                },
            )
            group["count"] += 1
            group["tiers"][item.tier.value] = group["tiers"].get(item.tier.value, 0) + 1
            group["namespaces"][item.namespace.value] = group["namespaces"].get(item.namespace.value, 0) + 1
            for tag in item.tags:
                tag_text = str(tag).strip()
                if tag_text:
                    group["top_tags"][tag_text] = group["top_tags"].get(tag_text, 0) + 1
            if len(group["examples"]) < 3:
                group["examples"].append({"id": item.id, "content": item.content[:180], "tier": item.tier.value})
        rows = sorted(groups.values(), key=lambda row: (-int(row["count"]), str(row["community"])))
        for row in rows:
            row["top_tags"] = [tag for tag, _ in sorted(row["top_tags"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
        return {"count": len(rows), "communities": rows[: max(1, int(limit))]}

    @eidosian()
    def memory_graph(self, limit: int = 120) -> Dict[str, Any]:
        """Return a graph-friendly view of memory nodes and semantic/community links."""
        items = self.list_all()[: max(1, int(limit))]
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_edges: Set[tuple[str, str, str]] = set()
        for item in items:
            nodes.append(
                {
                    "id": item.id,
                    "label": item.content[:72],
                    "community": str(item.metadata.get("community") or "unclassified"),
                    "tier": item.tier.value,
                    "namespace": item.namespace.value,
                    "tags": sorted(item.tags)[:8],
                }
            )
        by_id = {item.id: item for item in items}
        for item in items:
            for linked_id in sorted(item.linked_memories):
                if linked_id not in by_id:
                    continue
                key = tuple(sorted((item.id, linked_id)) + ["linked"])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({"source": item.id, "target": linked_id, "rel_type": "linked"})
        community_buckets: Dict[str, List[str]] = {}
        for item in items:
            community = str(item.metadata.get("community") or "")
            if not community:
                continue
            community_buckets.setdefault(community, []).append(item.id)
        for members in community_buckets.values():
            for idx in range(len(members) - 1):
                src = members[idx]
                dst = members[idx + 1]
                key = tuple(sorted((src, dst)) + ["community"])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({"source": src, "target": dst, "rel_type": "community"})
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "community_count": len(community_buckets),
            },
        }

    def _apply_enrichment(self, item: TieredMemoryItem, *, use_llm: Optional[bool]) -> bool:
        enriched_tags, enriched_metadata = self._heuristic_enrichment(item)
        if bool(self.llm_enrichment if use_llm is None else use_llm):
            llm_metadata = self._llm_enrichment(item)
            if llm_metadata:
                llm_tags = {
                    self._normalize_tag(tag) for tag in llm_metadata.get("tags") or [] if self._normalize_tag(tag)
                }
                enriched_tags.update(llm_tags)
                enriched_metadata = self._merge_metadata(enriched_metadata, llm_metadata)
                enriched_metadata["tag_origin"] = "llm_assisted"
        changed = False
        if not enriched_tags.issubset(item.tags):
            item.tags.update(enriched_tags)
            changed = True
        merged = self._merge_metadata(item.metadata, enriched_metadata)
        if merged != item.metadata:
            item.metadata = merged
            changed = True
        if item.embedding is None and self.embedder:
            try:
                item.embedding = self.embedder.embed_text(item.content)
                changed = True
            except Exception:
                pass
        return changed

    def _heuristic_enrichment(self, item: TieredMemoryItem) -> tuple[Set[str], Dict[str, Any]]:
        keywords = self._extract_keywords(item.content)
        domains = self._infer_domains(item.content, set(item.tags), item.namespace, item.memory_type)
        tags = {
            self._normalize_tag(tag)
            for tag in (
                list(item.tags) + keywords + domains + [item.tier.value, item.namespace.value, item.memory_type.value]
            )
            if self._normalize_tag(tag)
        }
        title = item.content.strip().splitlines()[0][:96] if item.content.strip() else ""
        community = self._community_label(item.namespace, domains, keywords)
        metadata = {
            "content_hash": self._content_hash(item.content),
            "keywords": keywords,
            "domains": domains,
            "community": community,
            "title": title,
            "summary": item.content.strip().replace("\n", " ")[:180],
            "tag_origin": str(item.metadata.get("tag_origin") or "heuristic"),
            "vector_ready": bool(item.embedding),
        }
        return tags, metadata

    def _llm_enrichment(self, item: TieredMemoryItem) -> Dict[str, Any]:
        try:
            from eidos_mcp.config.models import ModelConfig
        except Exception:
            return {}
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "domains": {"type": "array", "items": {"type": "string"}},
                "community": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["tags", "keywords", "domains", "community", "summary"],
        }
        prompt = "\n".join(
            [
                "Return structured enrichment for this memory.",
                f"Tier: {item.tier.value}",
                f"Namespace: {item.namespace.value}",
                f"Type: {item.memory_type.value}",
                "Memory:",
                item.content[:2000],
            ]
        )
        try:
            payload = ModelConfig().generate_payload(
                prompt,
                model=self.llm_model,
                max_tokens=320,
                temperature=0.1,
                thinking_mode=self.llm_thinking_mode,
                timeout=180.0,
                format=schema,
            )
            response = str(payload.get("response") or "").strip()
            if not response:
                return {}
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                return {}
            return {
                "keywords": [self._normalize_tag(x) for x in parsed.get("keywords") or [] if self._normalize_tag(x)],
                "domains": [self._normalize_tag(x) for x in parsed.get("domains") or [] if self._normalize_tag(x)],
                "tags": [self._normalize_tag(x) for x in parsed.get("tags") or [] if self._normalize_tag(x)],
                "community": self._normalize_tag(parsed.get("community")) or "",
                "summary": str(parsed.get("summary") or "").strip()[:220],
            }
        except Exception:
            return {}

    def _extract_keywords(self, text: str, limit: int = 8) -> List[str]:
        counts: Dict[str, int] = {}
        for token in self._WORD_RE.findall(str(text or "").lower()):
            norm = self._normalize_tag(token)
            if not norm or norm.isdigit() or len(norm) < 4:
                continue
            counts[norm] = counts.get(norm, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, int(limit))]]

    def _infer_domains(
        self,
        content: str,
        tags: Set[str],
        namespace: MemoryNamespace,
        memory_type: MemoryType,
    ) -> List[str]:
        hay = {self._normalize_tag(token) for token in self._WORD_RE.findall(str(content or "").lower())}
        hay.update({self._normalize_tag(tag) for tag in tags})
        hay.add(self._normalize_tag(namespace.value))
        hay.add(self._normalize_tag(memory_type.value))
        matched: List[str] = []
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            if hay.intersection(keywords):
                matched.append(domain)
        if not matched:
            matched.append(self._normalize_tag(namespace.value) or "general")
        return matched[:4]

    def _community_label(self, namespace: MemoryNamespace, domains: List[str], keywords: List[str]) -> str:
        primary_domain = domains[0] if domains else "general"
        secondary = keywords[0] if keywords else primary_domain
        return f"{self._normalize_tag(namespace.value)}:{self._normalize_tag(primary_domain)}:{self._normalize_tag(secondary)}".strip(
            ":"
        )

    def _normalize_tag(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        clean = []
        for ch in text:
            clean.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
        normalized = "".join(clean).strip("_-")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized[:48]

    def _content_hash(self, text: str) -> str:
        import hashlib

        return hashlib.sha256(str(text or "").encode("utf-8", "replace")).hexdigest()[:16]

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

    def _memory_similarity(self, item_a: TieredMemoryItem, item_b: TieredMemoryItem) -> float:
        """Similarity for compression clustering (embedding-first, lexical fallback)."""
        if item_a.embedding and item_b.embedding and len(item_a.embedding) == len(item_b.embedding):
            return self._cosine_similarity(item_a.embedding, item_b.embedding)
        tokens_a = self._tokenize(item_a.content)
        tokens_b = self._tokenize(item_b.content)
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union == 0:
            return 0.0
        return overlap / union

    def _semantic_cluster_summary(self, cluster: List[TieredMemoryItem]) -> str:
        """Build deterministic summary text for a semantic compression cluster."""
        ordered = sorted(cluster, key=lambda item: item.created_at)
        preview = [item.content.strip().replace("\n", " ")[:220] for item in ordered[:12]]
        return "\n".join(
            [
                f"Semantic compression summary for {len(cluster)} related memories:",
                *[f"- {line}" for line in preview if line],
            ]
        )

    def _tokenize(self, text: str) -> Set[str]:
        return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()}

    def _persist_tier(self, tier: MemoryTier) -> None:
        """Save a specific tier to disk."""
        path = self.persistence_dir / f"{tier.value}.json"
        data = [item.to_dict() for item in self.tiers[tier].values()]
        self._atomic_write_json(path, data)

    def _load(self) -> None:
        """Load all tiers from disk."""
        self.tiers = self._read_all_tiers()

    @eidosian()
    def save_all(self) -> None:
        """Persist all tiers to disk."""
        with self._mutation_lock():
            self._merge_disk_state_locked()
            for tier in MemoryTier:
                self._persist_tier(tier)

    @contextmanager
    def _mutation_lock(self):
        with self._thread_lock:
            if self._lock_depth == 0:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                self._lock_handle = open(self._lock_path, "a+", encoding="utf-8")
                if fcntl is not None:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
        try:
            yield
        finally:
            with self._thread_lock:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_handle is not None:
                    if fcntl is not None:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    self._lock_handle.close()
                    self._lock_handle = None

    def _read_all_tiers(self) -> Dict[MemoryTier, Dict[str, TieredMemoryItem]]:
        tiers: Dict[MemoryTier, Dict[str, TieredMemoryItem]] = {tier: {} for tier in MemoryTier}
        for tier in MemoryTier:
            path = self.persistence_dir / f"{tier.value}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item_data in data:
                    item = TieredMemoryItem.from_dict(item_data)
                    tiers[tier][item.id] = item
            except Exception:
                continue
        return tiers

    def _reload_from_disk_locked(self) -> None:
        self.tiers = self._read_all_tiers()

    def _merge_disk_state_locked(self) -> None:
        disk_tiers = self._read_all_tiers()
        merged: Dict[MemoryTier, Dict[str, TieredMemoryItem]] = {tier: {} for tier in MemoryTier}
        for tier in MemoryTier:
            merged[tier].update(disk_tiers[tier])
            for item_id, local_item in self.tiers[tier].items():
                disk_item = merged[tier].get(item_id)
                if disk_item is None:
                    merged[tier][item_id] = local_item
                else:
                    merged[tier][item_id] = self._merge_items(disk_item, local_item)
        self.tiers = merged

    def _remember_locked(
        self,
        content: str,
        tier: MemoryTier,
        namespace: MemoryNamespace,
        memory_type: MemoryType,
        importance: float,
        tags: Optional[Set[str]],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[List[float]],
    ) -> str:
        """Store or merge a memory while holding the persistence lock."""
        normalized_tags = set(tags or set())
        normalized_metadata = dict(metadata or {})
        preview_item = TieredMemoryItem(
            content=content,
            tier=tier,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            ttl_seconds=self.TIER_TTL.get(tier),
            tags=normalized_tags,
            metadata=normalized_metadata,
        )
        enriched_tags, enriched_metadata = self._heuristic_enrichment(preview_item)
        normalized_tags.update(enriched_tags)
        normalized_metadata = self._merge_metadata(normalized_metadata, enriched_metadata)
        existing = self._find_duplicate_memory(
            content=content,
            tier=tier,
            namespace=namespace,
            memory_type=memory_type,
            metadata=normalized_metadata,
        )
        if existing is not None:
            existing.tags.update(normalized_tags)
            existing.metadata = self._merge_metadata(existing.metadata, normalized_metadata)
            existing.importance = max(existing.importance, importance)
            existing.last_accessed = datetime.now()
            if existing.embedding is None and embedding:
                existing.embedding = embedding
            self._apply_enrichment(existing, use_llm=False)
            self._persist_tier(existing.tier)
            self._upsert_vector_store(existing)
            return existing.id

        item = TieredMemoryItem(
            content=content,
            tier=tier,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            ttl_seconds=self.TIER_TTL.get(tier),
            tags=normalized_tags,
            metadata=normalized_metadata,
        )
        self.tiers[tier][item.id] = item
        self._persist_tier(tier)
        self._upsert_vector_store(item)
        return item.id

    def _find_duplicate_memory(
        self,
        content: str,
        tier: MemoryTier,
        namespace: MemoryNamespace,
        memory_type: MemoryType,
        metadata: Dict[str, Any],
    ) -> Optional[TieredMemoryItem]:
        signature = self._memory_signature(
            content=content,
            tier=tier,
            namespace=namespace,
            memory_type=memory_type,
            metadata=metadata,
        )
        for item in self.tiers[tier].values():
            if (
                self._memory_signature(
                    content=item.content,
                    tier=item.tier,
                    namespace=item.namespace,
                    memory_type=item.memory_type,
                    metadata=item.metadata,
                )
                == signature
            ):
                return item
        return None

    def _memory_signature(
        self,
        content: str,
        tier: MemoryTier,
        namespace: MemoryNamespace,
        memory_type: MemoryType,
        metadata: Dict[str, Any],
    ) -> str:
        derived_keys = {
            "content_hash",
            "keywords",
            "domains",
            "community",
            "title",
            "summary",
            "tag_origin",
            "vector_ready",
            "enriched_at",
        }
        stable_metadata = {key: value for key, value in metadata.items() if key not in derived_keys}
        return json.dumps(
            {
                "content": content.strip(),
                "tier": tier.value,
                "namespace": namespace.value,
                "memory_type": memory_type.value,
                "metadata": self._normalize_json_value(stable_metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _merge_items(self, disk_item: TieredMemoryItem, local_item: TieredMemoryItem) -> TieredMemoryItem:
        merged = TieredMemoryItem(
            id=local_item.id,
            content=local_item.content or disk_item.content,
            created_at=min(disk_item.created_at, local_item.created_at),
            last_accessed=max(disk_item.last_accessed, local_item.last_accessed),
            access_count=max(disk_item.access_count, local_item.access_count),
            tier=local_item.tier,
            namespace=local_item.namespace,
            memory_type=local_item.memory_type,
            embedding=local_item.embedding or disk_item.embedding,
            metadata=self._merge_metadata(disk_item.metadata, local_item.metadata),
            importance=max(disk_item.importance, local_item.importance),
            ttl_seconds=local_item.ttl_seconds if local_item.ttl_seconds is not None else disk_item.ttl_seconds,
            tags=set(disk_item.tags) | set(local_item.tags),
            linked_memories=set(disk_item.linked_memories) | set(local_item.linked_memories),
        )
        return merged

    def _merge_metadata(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = self._merge_metadata(existing, value)
            elif isinstance(existing, list) and isinstance(value, list):
                merged[key] = list(dict.fromkeys(existing + value))
            elif isinstance(existing, set) and isinstance(value, set):
                merged[key] = sorted(existing | value)
            elif existing in (None, "", []):
                merged[key] = value
            elif value not in (None, "", []):
                merged[key] = value
        return merged

    def _normalize_json_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._normalize_json_value(val) for key, val in sorted(value.items())}
        if isinstance(value, set):
            return [self._normalize_json_value(val) for val in sorted(value)]
        if isinstance(value, (list, tuple)):
            return [self._normalize_json_value(val) for val in value]
        return value

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _upsert_vector_store(self, item: TieredMemoryItem) -> None:
        if self.vector_store is None or not item.embedding:
            return
        self.vector_store.upsert(
            item.id,
            item.embedding,
            text=item.content,
            metadata={
                "tier": item.tier.value,
                "namespace": item.namespace.value,
                "memory_type": item.memory_type.value,
                "importance": item.importance,
                "tags": sorted(item.tags),
                "community": str(item.metadata.get("community") or ""),
                "keywords": list(item.metadata.get("keywords") or []),
                "domains": list(item.metadata.get("domains") or []),
            },
        )
