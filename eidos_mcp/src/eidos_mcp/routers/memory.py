from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List, Optional

from ..core import tool
from ..transactions import (
    begin_transaction,
    find_latest_transaction_for_path,
    load_transaction,
)
from ..forge_loader import ensure_forge_import
from ..embeddings import SimpleEmbedder

ensure_forge_import("memory_forge")

try:
    from memory_forge import MemoryConfig, MemoryForge
except Exception:  # pragma: no cover - fallback for missing deps
    MemoryConfig = None
    MemoryForge = None


FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))
MEMORY_PATH = Path(os.environ.get("EIDOS_MEMORY_PATH", FORGE_DIR / "memory_data.json"))

_embedder = SimpleEmbedder()


class _SimpleMemoryItem:
    def __init__(self, item_id: str, content: str):
        self.id = item_id
        self.content = content


class _SimpleMemoryStore:
    def __init__(self, path: Path, embedder: SimpleEmbedder):
        self.path = path
        self.embedder = embedder
        self.data: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def remember(self, content: str, metadata: Optional[dict] = None) -> str:
        item_id = str(uuid.uuid4())
        embedding = self.embedder.embed_text(content)
        self.data.append(
            {"id": item_id, "content": content, "embedding": embedding, "metadata": metadata or {}}
        )
        self._save()
        return item_id

    def recall(self, query: str, limit: int = 5) -> List[_SimpleMemoryItem]:
        q_vec = self.embedder.embed_text(query)
        q_norm = sum(v * v for v in q_vec) ** 0.5
        scored = []
        for entry in self.data:
            d_vec = entry.get("embedding") or []
            d_norm = sum(v * v for v in d_vec) ** 0.5
            if not d_vec or q_norm == 0.0 or d_norm == 0.0:
                score = 0.0
            else:
                score = sum(a * b for a, b in zip(q_vec, d_vec)) / (q_norm * d_norm)
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            _SimpleMemoryItem(item_id=e["id"], content=e["content"]) for _, e in scored[:limit]
        ]

    def delete(self, item_id: str) -> bool:
        before = len(self.data)
        self.data = [entry for entry in self.data if entry.get("id") != item_id]
        if len(self.data) != before:
            self._save()
            return True
        return False

    def clear(self) -> None:
        self.data = []
        self._save()

    def count(self) -> int:
        return len(self.data)


if MemoryForge and MemoryConfig:
    memory = MemoryForge(
        config=MemoryConfig(
            episodic={"connection_string": str(MEMORY_PATH), "type": "json"}
        ),
        embedder=_embedder,
    )
else:
    memory = _SimpleMemoryStore(MEMORY_PATH, _embedder)


@tool(
    name="memory_add",
    description="Add a new memory to episodic storage.",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "is_fact": {"type": "boolean"},
            "key": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["content"],
    },
)
def memory_add(
    content: str,
    is_fact: bool = False,
    key: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Add a new memory (episodic or semantic)."""
    txn = begin_transaction("memory_add", [MEMORY_PATH])
    try:
        meta = {"is_fact": is_fact}
        if key:
            meta["key"] = key
        if metadata:
            meta.update(metadata)
        mid = memory.remember(content, metadata=meta)
        txn.commit()
        return f"Memory added with ID: {mid} ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error storing memory: {exc}"


@tool(
    description="Retrieve memories relevant to a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
)
def memory_retrieve(query: str, limit: int = 5) -> str:
    """Search for relevant memories by query."""
    try:
        results = memory.recall(query, limit=limit)
        payload = [
            {"id": item.id, "content": item.content} for item in results
        ]
        return json.dumps(payload, indent=2)
    except Exception as exc:
        return f"Error retrieving memory: {exc}"


@tool(
    name="memory_delete",
    description="Delete a memory item by id.",
)
def memory_delete(item_id: str) -> str:
    """Delete a memory item by id."""
    backend = memory.episodic if hasattr(memory, "episodic") else memory
    with begin_transaction("memory_delete", [MEMORY_PATH]) as txn:
        deleted = backend.delete(item_id)
        if not deleted:
            txn.rollback("no-op: not found")
            return "No-op: Not found"
        return f"Deleted ({txn.id})"


@tool(
    name="memory_clear",
    description="Clear the episodic memory store.",
)
def memory_clear() -> str:
    """Clear the episodic memory store."""
    backend = memory.episodic if hasattr(memory, "episodic") else memory
    count = backend.count() if hasattr(backend, "count") else None
    if count == 0:
        return "No-op: Memory already empty"
    with begin_transaction("memory_clear", [MEMORY_PATH]) as txn:
        backend.clear()
        if hasattr(backend, "count") and backend.count() != 0:
            txn.rollback("verification_failed: not_empty")
            return f"Error: Verification failed; rolled back ({txn.id})"
        return f"Memory cleared ({txn.id})"


@tool(
    name="memory_stats",
    description="Return memory store statistics.",
)
def memory_stats() -> str:
    """Return memory store statistics."""
    try:
        if hasattr(memory, "stats"):
            stats = memory.stats()
        elif hasattr(memory, "count"):
            stats = {"episodic_count": memory.count()}
        else:
            stats = {}
        return json.dumps(stats, indent=2)
    except Exception as exc:
        return f"Error retrieving memory stats: {exc}"


@tool(
    name="memory_snapshot",
    description="Create a snapshot of the episodic memory store.",
    parameters={"type": "object", "properties": {}},
)
def memory_snapshot() -> str:
    """Create a snapshot of the episodic memory store."""
    txn = begin_transaction("memory_snapshot", [MEMORY_PATH])
    txn.commit("snapshot")
    return f"Snapshot created ({txn.id})"


@tool(
    name="memory_restore",
    description="Restore episodic memory from a snapshot transaction.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
    },
)
def memory_restore(transaction_id: Optional[str] = None) -> str:
    """Restore episodic memory from a snapshot transaction."""
    txn_id = transaction_id or find_latest_transaction_for_path(MEMORY_PATH)
    if not txn_id:
        return "Error: No transaction found for memory store"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("memory_restore")
    try:
        backend = memory.episodic if hasattr(memory, "episodic") else memory
        if hasattr(backend, "_load"):
            backend._load()
    except Exception:
        pass
    return f"Memory restored ({txn_id})"
