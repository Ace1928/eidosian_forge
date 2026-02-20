from eidosian_core import eidosian

"""
ChromaDB Backend for Vector Storage.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

import chromadb

from ..core.interfaces import MemoryItem, MemoryType, StorageBackend


class ChromaBackend(StorageBackend):
    def __init__(self, collection_name: str, persist_path: str):
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as exc:
            if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
                raise RuntimeError(
                    "Failed to initialize Chroma on Termux. "
                    "If you see illegal-instruction errors, rebuild hnsw wheels with "
                    "HNSWLIB_NO_NATIVE=1 in eidosian_venv."
                ) from exc
            raise

    @eidosian()
    def add(self, item: MemoryItem) -> bool:
        if not item.embedding:
            raise ValueError("ChromaBackend requires embeddings for items.")

        # Flatten metadata for Chroma compatibility (no nested dicts)
        meta = {k: str(v) for k, v in item.metadata.items()}
        meta["created_at"] = item.created_at.isoformat()
        meta["type"] = item.type.value
        meta["importance"] = item.importance

        self.collection.add(documents=[item.content], embeddings=[item.embedding], metadatas=[meta], ids=[item.id])
        return True

    @eidosian()
    def get(self, item_id: str) -> Optional[MemoryItem]:
        res = self.collection.get(ids=[item_id], include=["metadatas", "documents", "embeddings"])
        if not res["ids"]:
            return None

        return self._map_result_to_item(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["embeddings"][0] if res["embeddings"] is not None else None,
        )

    @eidosian()
    def search(self, query_embedding: List[float], limit: int = 10, filters: Optional[Dict] = None) -> List[MemoryItem]:
        res = self.collection.query(query_embeddings=[query_embedding], n_results=limit, where=filters)

        items = []
        if res["ids"]:
            for i, doc_id in enumerate(res["ids"][0]):
                item = self._map_result_to_item(
                    doc_id,
                    res["documents"][0][i],
                    res["metadatas"][0][i],
                    None,  # Query doesn't return embeddings by default for speed
                )
                items.append(item)
        return items

    @eidosian()
    def delete(self, item_id: str) -> bool:
        self.collection.delete(ids=[item_id])
        return True

    @eidosian()
    def count(self) -> int:
        return self.collection.count()

    @eidosian()
    def clear(self) -> None:
        # Chroma doesn't have a clear, so delete all
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    def _map_result_to_item(self, id: str, doc: str, meta: Dict, emb: Optional[List[float]]) -> MemoryItem:
        return MemoryItem(
            id=id,
            content=doc,
            created_at=datetime.fromisoformat(meta.get("created_at", datetime.now().isoformat())),
            type=MemoryType(meta.get("type", "episodic")),
            importance=float(meta.get("importance", 1.0)),
            metadata=meta,
            embedding=emb,
        )
