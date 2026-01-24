from eidosian_core import eidosian
"""
Simple JSON Backend for testing and portability.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict
from ..core.interfaces import StorageBackend, MemoryItem
import numpy as np

class JsonBackend(StorageBackend):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: Dict[str, dict] = {}
        if self.file_path.exists():
            self._load()

    def _load(self):
        try:
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def _save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2)

    @eidosian()
    def add(self, item: MemoryItem) -> bool:
        self.data[item.id] = item.to_dict()
        if item.embedding:
            self.data[item.id]["embedding"] = item.embedding
        self._save()
        return True

    @eidosian()
    def get(self, item_id: str) -> Optional[MemoryItem]:
        d = self.data.get(item_id)
        if not d: return None
        # Reconstruct (simplified)
        item = MemoryItem(content=d["content"], id=d["id"])
        # ... fill other fields ...
        return item

    @eidosian()
    def search(self, query_embedding: List[float], limit: int = 10, filters: Optional[Dict] = None) -> List[MemoryItem]:
        # Naive cosine similarity
        results = []
        q_vec = np.array(query_embedding)
        
        for mid, d in self.data.items():
            if "embedding" not in d: continue
            d_vec = np.array(d["embedding"])
            score = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            results.append((score, d))
            
        results.sort(key=lambda x: x[0], reverse=True)
        
        items = []
        for score, d in results[:limit]:
            items.append(MemoryItem(content=d["content"], id=d["id"])) # Simplified
        return items

    @eidosian()
    def delete(self, item_id: str) -> bool:
        if item_id in self.data:
            del self.data[item_id]
            self._save()
            return True
        return False

    @eidosian()
    def count(self) -> int:
        return len(self.data)

    @eidosian()
    def clear(self) -> None:
        self.data = {}
        self._save()
