from eidosian_core import eidosian
"""
Librarian - Manages the Code Snippet Database.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib

class CodeLibrarian:
    """
    Stores and retrieves code snippets.
    Uses a simple JSON store for now, intended to upgrade to Vector DB.
    """
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.db: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            with open(self.storage_path, "r") as f:
                return json.load(f)
        return {"snippets": {}}

    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.db, f, indent=2)

    @eidosian()
    def add_snippet(self, code: str, metadata: Dict[str, Any]) -> str:
        """Add a snippet to the library."""
        # Create a deterministic ID based on content
        snippet_id = hashlib.sha256(code.encode("utf-8")).hexdigest()
        
        self.db["snippets"][snippet_id] = {
            "code": code,
            "metadata": metadata,
            "tags": metadata.get("tags", [])
        }
        self._save()
        return snippet_id

    @eidosian()
    def get_snippet(self, snippet_id: str) -> Optional[Dict[str, Any]]:
        return self.db["snippets"].get(snippet_id)

    @eidosian()
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Naive text search."""
        results = []
        for sid, data in self.db["snippets"].items():
            if query in data["code"] or query in str(data["metadata"]):
                results.append(data)
        return results
