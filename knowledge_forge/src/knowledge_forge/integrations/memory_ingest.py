"""
Memory -> Knowledge ingestion bridge.
"""
from __future__ import annotations

from eidosian_core import eidosian

from pathlib import Path
from typing import Dict, Any, List
import json

from knowledge_forge.core.graph import KnowledgeForge


class MemoryIngestor:
    """
    Ingest memory artifacts into KnowledgeForge.
    Supports the memory_data.json format used by EIDOS.
    """
    def __init__(self, knowledge_path: Path):
        self.knowledge = KnowledgeForge(persistence_path=knowledge_path)

    @eidosian()
    def ingest_memory_file(self, memory_path: Path, tags: List[str] | None = None) -> Dict[str, Any]:
        if not memory_path.exists():
            return {"success": False, "error": "memory file not found"}
        data = json.loads(memory_path.read_text(encoding="utf-8"))
        count = 0
        for _, item in data.items():
            if isinstance(item, list):
                for entry in item:
                    content = entry.get("content", "")
                    metadata = entry.get("metadata", {})
                    self.knowledge.add_knowledge(
                        content,
                        tags=(tags or []) + ["memory_ingest"],
                        metadata={"memory_id": entry.get("id") or entry.get("key"), **metadata},
                    )
                    count += 1
                continue

            content = item.get("content", "")
            metadata = item.get("metadata", {})
            self.knowledge.add_knowledge(
                content,
                tags=(tags or []) + ["memory_ingest"],
                metadata={"memory_id": item.get("id"), **metadata},
            )
            count += 1
        return {"success": True, "count": count}
