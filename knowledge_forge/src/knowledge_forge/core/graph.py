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

    def _build_rdf_graph(self) -> Any:
        """Build an rdflib graph snapshot from the in-memory knowledge graph."""
        try:
            from rdflib import RDF, Graph, Literal, Namespace, URIRef
        except Exception as exc:
            raise RuntimeError("RDF support requires rdflib. Install knowledge_forge[rdf].") from exc

        graph = Graph()
        eidos_ns = Namespace("https://eidosian.dev/knowledge#")
        graph.bind("eidos", eidos_ns)

        concept_index: Dict[str, Set[str]] = {}
        for concept, node_ids in self.concept_map.items():
            concept_index[concept] = set(node_ids)

        def node_uri(node_id: str):
            return URIRef(f"urn:eidos:knowledge:node:{node_id}")

        for node in self.nodes.values():
            subject = node_uri(node.id)
            graph.add((subject, RDF.type, eidos_ns.KnowledgeNode))
            graph.add((subject, eidos_ns.content, Literal(str(node.content))))
            if node.metadata:
                graph.add((subject, eidos_ns.metadataJson, Literal(json.dumps(node.metadata, sort_keys=True))))
            for tag in sorted(node.tags):
                graph.add((subject, eidos_ns.tag, Literal(tag)))
            for concept, node_ids in concept_index.items():
                if node.id in node_ids:
                    graph.add((subject, eidos_ns.hasConcept, Literal(concept)))
            for linked_id in sorted(node.links):
                graph.add((subject, eidos_ns.linkedTo, node_uri(linked_id)))
        return graph

    def _import_rdf_graph(self, graph: Any, *, merge: bool = False) -> Dict[str, Any]:
        """Load graph state from an rdflib Graph object."""
        try:
            from rdflib import RDF, Namespace
        except Exception as exc:
            raise RuntimeError("RDF support requires rdflib. Install knowledge_forge[rdf].") from exc

        eidos_ns = Namespace("https://eidosian.dev/knowledge#")

        if not merge:
            self.nodes = {}
            self.concept_map = {}

        def extract_node_id(subject: Any) -> str:
            text = str(subject)
            marker = "urn:eidos:knowledge:node:"
            if marker in text:
                return text.split(marker, 1)[1]
            return text.rsplit(":", 1)[-1]

        subjects = set(graph.subjects(RDF.type, eidos_ns.KnowledgeNode))
        subjects.update(graph.subjects(eidos_ns.content, None))

        imported = 0
        for subject in subjects:
            node_id = extract_node_id(subject)
            content_literal = next(iter(graph.objects(subject, eidos_ns.content)), None)
            if content_literal is None:
                continue
            metadata: Dict[str, Any] = {}
            metadata_literal = next(iter(graph.objects(subject, eidos_ns.metadataJson)), None)
            if metadata_literal is not None:
                try:
                    metadata = json.loads(str(metadata_literal))
                except Exception:
                    metadata = {}

            node = KnowledgeNode(str(content_literal), metadata)
            node.id = node_id
            node.tags = {str(tag) for tag in graph.objects(subject, eidos_ns.tag)}
            self.nodes[node.id] = node
            imported += 1

        self.concept_map = {}
        for subject in subjects:
            source_id = extract_node_id(subject)
            if source_id not in self.nodes:
                continue
            source = self.nodes[source_id]
            for concept in graph.objects(subject, eidos_ns.hasConcept):
                concept_name = str(concept)
                self.concept_map.setdefault(concept_name, []).append(source_id)
            for target in graph.objects(subject, eidos_ns.linkedTo):
                target_id = extract_node_id(target)
                if target_id in self.nodes:
                    source.links.add(target_id)

        if self.persistence_path:
            self.save()

        return {
            "imported_nodes": imported,
            "concept_count": len(self.concept_map),
        }

    @eidosian()
    def export_rdf(self, output_path: Union[str, Path], format: str = "turtle") -> Dict[str, Any]:
        """
        Export knowledge graph as RDF using rdflib.

        Requires `rdflib` optional dependency.
        """
        graph = self._build_rdf_graph()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(output), format=format)
        return {
            "path": str(output),
            "format": format,
            "node_count": len(self.nodes),
            "triple_count": len(graph),
        }

    @eidosian()
    def import_rdf(
        self, input_path: Union[str, Path], format: Optional[str] = None, merge: bool = False
    ) -> Dict[str, Any]:
        """
        Import knowledge graph from RDF using rdflib.

        Requires `rdflib` optional dependency.
        """
        try:
            from rdflib import Graph
        except Exception as exc:
            raise RuntimeError("RDF import requires rdflib. Install knowledge_forge[rdf].") from exc

        graph = Graph()
        graph.parse(str(input_path), format=format)
        loaded = self._import_rdf_graph(graph, merge=merge)

        return {
            "path": str(input_path),
            "format": format or "auto",
            **loaded,
        }

    @staticmethod
    def reason_rdf_graph(
        graph: Any,
        *,
        profile: str = "owlrl",
        include_axiomatic: bool = False,
        include_datatype_axioms: bool = False,
    ) -> Dict[str, Any]:
        """Run owlrl reasoning over an rdflib graph and mutate it in-place."""
        try:
            from owlrl import DeductiveClosure, OWLRL_Semantics, RDFS_Semantics
        except Exception as exc:
            raise RuntimeError("OWL reasoning requires owlrl. Install knowledge_forge[reasoning].") from exc

        profile_norm = str(profile or "owlrl").strip().lower()
        semantics_map = {
            "owlrl": OWLRL_Semantics,
            "rdfs": RDFS_Semantics,
        }
        semantics = semantics_map.get(profile_norm)
        if semantics is None:
            supported = ", ".join(sorted(semantics_map.keys()))
            raise ValueError(f"Unsupported reasoning profile: {profile_norm}. Supported: {supported}")

        triple_count_before = len(graph)
        closure = DeductiveClosure(
            semantics,
            axiomatic_triples=bool(include_axiomatic),
            datatype_axioms=bool(include_datatype_axioms),
        )
        closure.expand(graph)
        triple_count_after = len(graph)
        inferred_triples = max(0, int(triple_count_after) - int(triple_count_before))

        return {
            "profile": profile_norm,
            "triple_count_before": int(triple_count_before),
            "triple_count_after": int(triple_count_after),
            "inferred_triples": inferred_triples,
            "include_axiomatic": bool(include_axiomatic),
            "include_datatype_axioms": bool(include_datatype_axioms),
        }

    @eidosian()
    def reason_owl(
        self,
        *,
        profile: str = "owlrl",
        apply: bool = False,
        merge: bool = False,
        include_axiomatic: bool = False,
        include_datatype_axioms: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "turtle",
    ) -> Dict[str, Any]:
        """
        Run OWL/RDFS reasoning on the current knowledge graph RDF projection.

        Requires `owlrl` optional dependency.
        """
        graph = self._build_rdf_graph()
        report = self.reason_rdf_graph(
            graph,
            profile=profile,
            include_axiomatic=include_axiomatic,
            include_datatype_axioms=include_datatype_axioms,
        )

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            graph.serialize(destination=str(output), format=output_format)
            report["output_path"] = str(output)
            report["output_format"] = output_format

        if apply:
            imported = self._import_rdf_graph(graph, merge=merge)
            report["applied"] = True
            report["merge"] = bool(merge)
            report["imported_nodes"] = int(imported["imported_nodes"])
            report["concept_count"] = int(imported["concept_count"])
        else:
            report["applied"] = False
            report["merge"] = bool(merge)

        return report

    @eidosian()
    def visualize_pyvis(
        self,
        output_path: Union[str, Path],
        max_nodes: int = 200,
        height: str = "800px",
        width: str = "100%",
    ) -> Dict[str, Any]:
        """
        Export an interactive graph visualization using pyvis.

        Requires `pyvis` optional dependency.
        """
        try:
            from pyvis.network import Network
        except Exception as exc:
            raise RuntimeError("Pyvis visualization requires pyvis. Install knowledge_forge[viz].") from exc

        node_items = list(self.nodes.values())[: max(1, int(max_nodes))]
        included_ids = {node.id for node in node_items}
        network = Network(height=height, width=width, directed=False, notebook=False)
        network.barnes_hut()

        for node in node_items:
            title = str(node.content)
            tags = ", ".join(sorted(node.tags)) if node.tags else "none"
            network.add_node(
                node.id,
                label=node.id[:8],
                title=f"{title}\n\nTags: {tags}",
                group=(sorted(node.tags)[0] if node.tags else "untagged"),
            )

        edge_count = 0
        seen_edges: Set[tuple[str, str]] = set()
        for node in node_items:
            for link in node.links:
                if link not in included_ids:
                    continue
                edge = tuple(sorted((node.id, link)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                network.add_edge(node.id, link)
                edge_count += 1

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        network.write_html(str(output))
        return {
            "path": str(output),
            "node_count": len(node_items),
            "edge_count": edge_count,
            "max_nodes": int(max_nodes),
        }
