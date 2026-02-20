"""
Memory Broker: orchestrates multi-layer memory (working, episodic, semantic, procedural, self, user).

This broker is intentionally lightweight and file-backed for portability.
It integrates with MemoryForge for episodic/semantic storage and provides
structured lookup for self/user memory categories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian

from .interfaces import MemoryItem, MemoryType
from .main import MemoryForge


@dataclass
class MemoryEnvelope:
    """Structured memory record for broker-managed layers."""

    id: str
    created_at: str
    type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBroker:
    """
    Broker for layered memory. Uses MemoryForge for episodic/semantic storage,
    and maintains local JSON stores for self/user/working/procedural.
    """

    def __init__(self, data_dir: Path, forge: Optional[MemoryForge] = None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.forge = forge or MemoryForge()

        self._stores = {
            "self": self.data_dir / "self_memory.json",
            "user": self.data_dir / "user_memory.json",
            "working": self.data_dir / "working_memory.json",
            "procedural": self.data_dir / "procedural_memory.json",
        }
        for path in self._stores.values():
            if not path.exists():
                path.write_text(json.dumps({}, indent=2))

    def _load_store(self, store_path: Path) -> Dict[str, Any]:
        try:
            return json.loads(store_path.read_text())
        except Exception:
            return {}

    def _save_store(self, store_path: Path, data: Dict[str, Any]) -> None:
        store_path.write_text(json.dumps(data, indent=2))

    @eidosian()
    def remember_self(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self._remember_layer("self", content, metadata)

    @eidosian()
    def remember_user(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self._remember_layer("user", content, metadata)

    @eidosian()
    def remember_working(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self._remember_layer("working", content, metadata)

    @eidosian()
    def remember_procedural(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self._remember_layer("procedural", content, metadata)

    def _remember_layer(self, layer: str, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        store_path = self._stores[layer]
        store = self._load_store(store_path)
        item = MemoryItem(content=content, type=MemoryType.PROCEDURAL if layer == "procedural" else MemoryType.WORKING)
        envelope = MemoryEnvelope(
            id=item.id,
            created_at=datetime.now().isoformat(),
            type=layer,
            content=content,
            metadata=metadata or {},
        )
        store[item.id] = envelope.__dict__
        self._save_store(store_path, store)
        return item.id

    @eidosian()
    def remember_episodic(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.forge.remember(content, metadata=metadata)

    @eidosian()
    def recall_episodic(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[MemoryItem]:
        return self.forge.recall(query, limit=limit, filter_metadata=filters)

    @eidosian()
    def recall_layer(self, layer: str, limit: int = 10) -> List[MemoryEnvelope]:
        store_path = self._stores[layer]
        store = self._load_store(store_path)
        items = list(store.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return [MemoryEnvelope(**i) for i in items[:limit]]

    @eidosian()
    def stats(self) -> Dict[str, Any]:
        stats = {"forge": self.forge.stats()}
        for layer, path in self._stores.items():
            stats[f"{layer}_count"] = len(self._load_store(path))
        return stats
