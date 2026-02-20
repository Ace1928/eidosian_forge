import json
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from eidosian_core import eidosian


class KnowledgeNode:
    """A node in the knowledge graph with tags and bidirectional links."""

    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.tags: Set[str] = set(self.metadata.get("tags", []))
        self.links: Set[str] = set()

    @eidosian()
    def add_link(self, node_id: str):
        self.links.add(node_id)

    @eidosian()
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {**self.metadata, "tags": list(self.tags)},
            "links": list(self.links),
        }


class KnowledgeForge:
    """
    Manages a persistent graph of knowledge nodes.
    Supports concept mapping, tagging, and pathfinding.
    """

    def __init__(self, persistence_path: Optional[Union[str, Path]] = None):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.concept_map: Dict[str, List[str]] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None

        if self.persistence_path and self.persistence_path.exists():
            self.load()

    @eidosian()
    def add_knowledge(
        self,
        content: Any,
        concepts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeNode:
        """Add a node, map concepts/tags, and persist."""
        node = KnowledgeNode(content, metadata)
        if tags:
            node.tags.update(tags)
        self.nodes[node.id] = node

        if concepts:
            for concept in concepts:
                if concept not in self.concept_map:
                    self.concept_map[concept] = []
                self.concept_map[concept].append(node.id)

        if self.persistence_path:
            self.save()
        return node

    @eidosian()
    def link_nodes(self, node_id_a: str, node_id_b: str):
        """Bidirectional link creation."""
        if node_id_a in self.nodes and node_id_b in self.nodes:
            self.nodes[node_id_a].add_link(node_id_b)
            self.nodes[node_id_b].add_link(node_id_a)
            if self.persistence_path:
                self.save()

    @eidosian()
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and remove all references."""
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        for node in self.nodes.values():
            if node_id in node.links:
                node.links.discard(node_id)
        for concept, node_ids in list(self.concept_map.items()):
            if node_id in node_ids:
                filtered = [nid for nid in node_ids if nid != node_id]
                if filtered:
                    self.concept_map[concept] = filtered
                else:
                    del self.concept_map[concept]
        if self.persistence_path:
            self.save()
        return True

    @eidosian()
    def get_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """Find nodes by tag."""
        return [n for n in self.nodes.values() if tag in n.tags]

    @eidosian()
    def get_by_concept(self, concept: str) -> List[KnowledgeNode]:
        """Retrieve all nodes associated with a concept."""
        node_ids = self.concept_map.get(concept, [])
        return [self.nodes[nid] for nid in node_ids]

    @eidosian()
    def get_related_nodes(self, node_id: str) -> List[KnowledgeNode]:
        """Get all nodes directly linked to the given node."""
        if node_id not in self.nodes:
            return []
        return [self.nodes[link_id] for link_id in self.nodes[node_id].links]

    @eidosian()
    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """BFS to find the shortest path between two nodes."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        queue = deque([[start_id]])
        visited = {start_id}

        while queue:
            path = queue.popleft()
            node_id = path[-1]

            if node_id == end_id:
                return path

            for neighbor in self.nodes[node_id].links:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return []

    @eidosian()
    def save(self):
        """Persist graph to disk."""
        data = {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}, "concept_map": self.concept_map}
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    @eidosian()
    def load(self):
        """Load graph from disk."""
        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
                for nid, n_data in data.get("nodes", {}).items():
                    node = KnowledgeNode(n_data["content"], n_data["metadata"])
                    node.id = n_data["id"]
                    node.links = set(n_data["links"])
                    node.tags = set(n_data["metadata"].get("tags", []))
                    self.nodes[node.id] = node
                self.concept_map = data.get("concept_map", {})
        except Exception:
            pass

    @eidosian()
    def search(self, query: str) -> List[KnowledgeNode]:
        results = []
        for node in self.nodes.values():
            if query.lower() in str(node.content).lower():
                results.append(node)
        return results

    @eidosian()
    def list_nodes(self, limit: int = 100) -> List[KnowledgeNode]:
        """List all nodes (up to limit)."""
        return list(self.nodes.values())[:limit]

    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Return knowledge graph statistics."""
        return {
            "node_count": len(self.nodes),
            "concept_count": len(self.concept_map),
            "concepts": list(self.concept_map.keys()),
        }
