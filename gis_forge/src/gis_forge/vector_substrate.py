from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import hnswlib
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("gis_forge.vector_substrate")


class VectorEntry(BaseModel):
    """A single entry in the vector substrate."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSubstrate:
    """
    A high-performance, production-quality vector storage engine built on hnswlib.
    Designed for the Eidosian ecosystem with support for persistence, schema extension,
    and efficient querying.
    """

    def __init__(self, storage_dir: Path, dim: int = 768, space: str = "cosine", max_elements: int = 100000):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.dim = dim
        self.space = space
        self.max_elements = max_elements

        self.index_path = self.storage_dir / "index.bin"
        self.metadata_path = self.storage_dir / "metadata.json"

        self.index = hnswlib.Index(space=self.space, dim=self.dim)

        # Mapping of integer internal IDs to UUID string IDs
        self.id_map: Dict[int, str] = {}
        # Mapping of UUID string IDs to internal IDs
        self.rev_id_map: Dict[str, int] = {}
        # Storage for actual metadata
        self.metadata_store: Dict[str, Dict[str, Any]] = {}

        self._load_or_init()

    def _load_or_init(self):
        """Initialize a new index or load from disk."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                # Load metadata first to know the current count
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                    self.id_map = {int(k): v for k, v in data.get("id_map", {}).items()}
                    self.rev_id_map = data.get("rev_id_map", {})
                    self.metadata_store = data.get("metadata_store", {})

                self.index.load_index(str(self.index_path), max_elements=self.max_elements)
                logger.info(f"Loaded vector index with {len(self.id_map)} elements.")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}. Re-initializing.")
                self._init_new_index()
        else:
            self._init_new_index()

    def _init_new_index(self):
        """Create a fresh index."""
        self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
        self.id_map = {}
        self.rev_id_map = {}
        self.metadata_store = {}
        logger.info(f"Initialized fresh vector index (dim={self.dim}, space={self.space}).")

    def save(self):
        """Persist index and metadata to disk."""
        self.index.save_index(str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(
                {"id_map": self.id_map, "rev_id_map": self.rev_id_map, "metadata_store": self.metadata_store},
                f,
                indent=2,
            )
        logger.info("Vector substrate persisted.")

    def add_items(self, entries: List[VectorEntry]):
        """Add multiple items to the index."""
        if not entries:
            return

        vectors = np.array([e.vector for e in entries], dtype="float32")

        # Generate internal IDs
        start_id = len(self.id_map)
        internal_ids = np.arange(start_id, start_id + len(entries))

        # Add to index
        self.index.add_items(vectors, internal_ids)

        # Update maps
        for i, entry in enumerate(entries):
            int_id = int(internal_ids[i])
            self.id_map[int_id] = entry.id
            self.rev_id_map[entry.id] = int_id
            self.metadata_store[entry.id] = entry.metadata

        self.save()

    def query(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Query for nearest neighbors."""
        if len(self.id_map) == 0:
            return []

        query_vec = np.array([vector], dtype="float32")
        labels, distances = self.index.knn_query(query_vec, k=k)

        results = []
        for label, dist in zip(labels[0], distances[0]):
            node_id = self.id_map.get(int(label))
            if node_id:
                results.append(
                    {"id": node_id, "distance": float(dist), "metadata": self.metadata_store.get(node_id, {})}
                )
        return results

    def get_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific ID."""
        return self.metadata_store.get(node_id)

    def get_vectors(self, node_ids: List[str]) -> List[List[float]]:
        """Retrieve actual vectors for a list of node IDs."""
        internal_ids = [self.rev_id_map[nid] for nid in node_ids if nid in self.rev_id_map]
        if not internal_ids:
            return []

        # hnswlib.get_items returns normalized vectors for cosine space
        vectors = self.index.get_items(internal_ids)
        return vectors.tolist()

    def update_metadata(self, node_id: str, metadata: Dict[str, Any]):
        """Update metadata for an existing entry."""
        if node_id in self.metadata_store:
            self.metadata_store[node_id].update(metadata)
            self.save()
            return True
        return False
