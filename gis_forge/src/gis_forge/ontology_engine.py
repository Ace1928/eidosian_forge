from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .vector_substrate import VectorEntry, VectorSubstrate

logger = logging.getLogger("gis_forge.ontology_engine")


class Node(BaseModel):
    """A structured node in the ontology."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # 'memory', 'fact', 'lexicon', 'code', 'axiom'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector: Optional[List[float]] = None


class Edge(BaseModel):
    """A directional relationship between nodes."""

    source: str
    target: str
    type: str  # 'is_a', 'part_of', 'derived_from', 'relates_to'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OntologyEngine:
    """
    High-level manager for the Unified Eidosian Ontology.
    Orchestrates the vector substrate and manages the graph topology.
    """

    def __init__(self, storage_dir: Path, dim: int = 768):
        self.storage_dir = Path(storage_dir)
        self.substrate = VectorSubstrate(self.storage_dir, dim=dim)

    def add_node(self, node: Node, vector: List[float]):
        """Ingest a node into the ontology."""
        entry = VectorEntry(
            id=node.id,
            vector=vector,
            metadata={
                "type": node.type,
                "content": node.content,
                **node.metadata,
                "_links": "[]",  # Initialize empty links
            },
        )
        self.substrate.add_items([entry])
        return node.id

    def link_nodes(self, source_id: str, target_id: str, link_type: str, metadata: Optional[Dict] = None):
        """Establish a link between two nodes."""
        source_meta = self.substrate.get_metadata(source_id)
        if not source_meta:
            logger.error(f"Source node {source_id} not found.")
            return False

        links = json.loads(source_meta.get("_links", "[]"))
        links.append({"target": target_id, "type": link_type, "metadata": metadata or {}})

        self.substrate.update_metadata(source_id, {"_links": json.dumps(links)})
        return True

    def semantic_search(self, query_vector: List[float], k: int = 5, node_type: Optional[str] = None):
        """Perform semantic search with optional type filtering."""
        raw_results = self.substrate.query(query_vector, k=k * 2)  # Get more to filter

        filtered = []
        for res in raw_results:
            if node_type and res["metadata"].get("type") != node_type:
                continue
            filtered.append(res)
            if len(filtered) >= k:
                break
        return filtered

    def get_node(self, node_id: str):
        """Retrieve a full node by ID."""
        meta = self.substrate.get_metadata(node_id)
        if meta:
            return Node(
                id=node_id,
                type=meta.get("type", "unknown"),
                content=meta.get("content", ""),
                metadata={k: v for k, v in meta.items() if k not in ["type", "content"]},
            )
        return None
