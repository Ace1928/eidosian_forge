from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian
from pydantic import BaseModel, Field

# Try to import vector backend (ChromaDB for now, easily swappable)
try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class OntologyNode(BaseModel):
    """A universal node in the Eidosian Ontology."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str  # 'memory', 'fact', 'lexicon', 'code', 'axiom'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OntologyEdge(BaseModel):
    """A multidimensional relationship between nodes."""

    source_id: str
    target_id: str
    edge_type: str  # 'is_a', 'derived_from', 'contradicts', 'related_to'
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UnifiedOntology:
    """
    The master vector-backed ontology manager for the Eidosian Forge.
    Manages the multidimensional graph network and coordinates lossless compression.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/ontology_db")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.collection = None

        if HAS_CHROMA:
            self.client = chromadb.PersistentClient(path=str(self.storage_path))
            self.collection = self.client.get_or_create_collection(name="eidosian_ontology")
        else:
            print("WARNING: ChromaDB not installed. Unified Ontology requires a vector backend.")

    @eidosian()
    def add_node(self, node: OntologyNode) -> str:
        """Add a universal node to the vectorized ontology."""
        if not self.collection:
            raise RuntimeError("Vector backend unavailable.")

        # We store edges and complex metadata in the ChromaDB metadata payload
        self.collection.add(
            documents=[node.content], metadatas=[{"node_type": node.node_type, **node.metadata}], ids=[node.id]
        )
        return node.id

    @eidosian()
    def search_ontology(
        self, query: str, n_results: int = 5, filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search across all dimensions of the ontology."""
        if not self.collection:
            raise RuntimeError("Vector backend unavailable.")

        where_clause = {"node_type": filter_type} if filter_type else None

        results = self.collection.query(query_texts=[query], n_results=n_results, where=where_clause)

        # Reconstruct structured response
        nodes = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                nodes.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                )
        return nodes

    @eidosian()
    def link_nodes(self, edge: OntologyEdge) -> None:
        """
        In a purely vector DB like Chroma, edges are often stored as metadata references
        or in a separate relational table. For the MVP, we update the source node's metadata
        to include the target ID, establishing the directional link.
        """
        if not self.collection:
            return

        try:
            source = self.collection.get(ids=[edge.source_id])
            if source and source["metadatas"]:
                meta = source["metadatas"][0]
                edges = json.loads(meta.get("_edges", "[]"))
                edges.append(edge.model_dump())
                meta["_edges"] = json.dumps(edges)

                self.collection.update(ids=[edge.source_id], metadatas=[meta])
        except Exception as e:
            print(f"Error linking nodes: {e}")
