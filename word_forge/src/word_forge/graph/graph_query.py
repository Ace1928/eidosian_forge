"""
Provides methods for querying graph information and structure.

Encapsulates logic for retrieving node/edge counts, finding related terms,
extracting subgraphs, getting node IDs, and generating graph summaries.
Adheres to Eidosian principles of modularity, clarity, and precision.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│    GraphQuery    │
    │ (Orchestrator)   │      │ (Information &   │
    └────────┬─────────┘      │ Structure Access)│
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │    NetworkX      │
    │  (Graph Object)  │
    └──────────────────┘
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
from eidosian_core import eidosian

# Import necessary components
from word_forge.exceptions import NodeNotFoundError
from word_forge.graph.graph_config import (
    GraphInfoDict,
    RelationshipDimension,
    RelType,
    Term,
    WordId,
    WordTupleDict,
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphQuery:
    """
    Provides methods to query and retrieve information from the graph.

    Offers functionalities like finding related terms, getting node/edge counts,
    looking up terms by ID, generating graph summaries, and extracting subgraphs.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphQuery with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    @eidosian()
    def get_node_id(self, term: Term) -> Optional[WordId]:
        """
        Retrieve the node ID for a given term (case-insensitive).

        Args:
            term: The term (word or phrase) to look up.

        Returns:
            Optional[WordId]: The integer ID of the node if found, else None.
        """
        return self.manager._term_to_id.get(term.lower())

    @eidosian()
    def get_related_terms(self, term: Term, rel_type: Optional[RelType] = None) -> List[Term]:
        """
        Find terms directly related to a given term in the graph.

        Args:
            term: The term for which to find related terms.
            rel_type: Optional filter to return only terms connected by a
                      specific relationship type (e.g., 'synonym').

        Returns:
            List[Term]: A list of related terms (strings).

        Raises:
            NodeNotFoundError: If the input term is not found in the graph.
        """
        start_node_id = self.get_node_id(term)
        if start_node_id is None:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph.")

        related_terms_list: List[Term] = []
        # Iterate through neighbors of the start node
        for neighbor_id in self.manager.g.neighbors(start_node_id):
            # Get edge data between start_node and neighbor
            # Note: For MultiGraph, this might return multiple edges. Assuming Graph for now.
            edge_data = self.manager.g.get_edge_data(start_node_id, neighbor_id)
            if not edge_data:  # Should not happen for neighbors, but safety check
                continue

            current_rel_type = edge_data.get("relationship")

            # Apply relationship type filter if provided
            if rel_type is None or (current_rel_type and current_rel_type.lower() == rel_type.lower()):
                # Get the term attribute from the neighbor node
                neighbor_term = self.manager.g.nodes[neighbor_id].get("term")
                if neighbor_term:
                    related_terms_list.append(neighbor_term)
                else:
                    self.logger.warning(f"Neighbor node {neighbor_id} of '{term}' is missing 'term' attribute.")

        self.logger.debug(f"Found {len(related_terms_list)} related terms for '{term}' (filter: {rel_type}).")
        return related_terms_list

    @eidosian()
    def get_node_count(self) -> int:
        """
        Get the total number of nodes (words) in the graph.

        Returns:
            int: The number of nodes.
        """
        return self.manager.g.number_of_nodes()

    @eidosian()
    def get_edge_count(self) -> int:
        """
        Get the total number of edges (relationships) in the graph.

        Returns:
            int: The number of edges.
        """
        return self.manager.g.number_of_edges()

    @eidosian()
    def get_term_by_id(self, word_id: WordId) -> Optional[Term]:
        """
        Retrieve the term associated with a given node ID.

        Args:
            word_id: The integer ID of the node.

        Returns:
            Optional[Term]: The term string if the node exists and has a 'term'
                            attribute, else None.
        """
        if word_id in self.manager.g.nodes():
            node_attrs = dict(self.manager.g.nodes(data=True))
            term_attr = node_attrs.get(word_id, {}).get("term")
            return str(term_attr) if term_attr is not None else None
        return None

    @eidosian()
    def get_graph_info(self) -> GraphInfoDict:
        """
        Retrieve detailed information about the graph structure and content.

        Provides counts, dimensions, sample nodes/relationships, and a list
        of unique relationship types present in the graph.

        Returns:
            GraphInfoDict: A dictionary containing graph metrics and samples.
        """
        nodes = self.get_node_count()
        edges = self.get_edge_count()

        # Sample nodes
        sample_node_ids = list(self.manager.g.nodes())[:5]
        sample_nodes_data: List[WordTupleDict] = []
        for node_id in sample_node_ids:
            node_data = self.manager.g.nodes[node_id]
            term = node_data.get("term", f"ID:{node_id}")  # Provide default if term missing
            sample_nodes_data.append({"id": node_id, "term": term})

        # Sample relationships (more complex to get terms efficiently)
        sample_relationships_data: List[Dict[str, str]] = []
        edge_count = 0
        for u, v, data in self.manager.g.edges(data=True):
            if edge_count >= 5:
                break
            term_u = self.manager.g.nodes[u].get("term", f"ID:{u}")
            term_v = self.manager.g.nodes[v].get("term", f"ID:{v}")
            rel_type = data.get("relationship", "unknown")
            sample_relationships_data.append({"source": term_u, "target": term_v, "type": rel_type})
            edge_count += 1

        # Get unique relationship types from stored counts
        relationship_types = sorted(list(self.manager._relationship_counts.keys()))

        info: GraphInfoDict = {
            "nodes": nodes,
            "edges": edges,
            "dimensions": self.manager.dimensions,
            "sample_nodes": sample_nodes_data,
            "sample_relationships": sample_relationships_data,
            "relationship_types": relationship_types,
        }
        return info

    @eidosian()
    def display_graph_summary(self) -> None:
        """
        Print a formatted summary of the graph's statistics to the console.

        Includes node/edge counts, dimensions, and counts of the most common
        relationship types.
        """
        info = self.get_graph_info()
        print("\n--- Graph Summary ---")
        print(f"Nodes: {info['nodes']}")
        print(f"Edges: {info['edges']}")
        print(f"Dimensions: {info['dimensions']}D")

        if self.manager._relationship_counts:
            print("\nRelationship Type Counts:")
            # Sort counts descending for display
            sorted_counts = sorted(
                self.manager._relationship_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for rel_type, count in sorted_counts[:10]:  # Show top 10
                print(f"  - {rel_type}: {count}")
            if len(sorted_counts) > 10:
                print(f"  ... and {len(sorted_counts) - 10} more types.")
        else:
            print("\nNo relationship types found.")
        print("---------------------\n")

    @eidosian()
    def get_subgraph(self, term: Term, depth: int = 1) -> nx.Graph:
        """
        Extract a subgraph centered around a specific term up to a given depth.

        Performs a breadth-first search from the term's node.

        Args:
            term: The central term for the subgraph.
            depth: The maximum distance (number of hops) from the central term.

        Returns:
            nx.Graph: A NetworkX graph object representing the subgraph.

        Raises:
            NodeNotFoundError: If the input term is not found in the graph.
            ValueError: If depth is negative.
        """
        if depth < 0:
            raise ValueError("Subgraph depth cannot be negative.")

        start_node_id = self.get_node_id(term)
        if start_node_id is None:
            raise NodeNotFoundError(f"Term '{term}' not found for subgraph extraction.")

        # Use ego_graph to find all nodes within the specified radius (depth)
        subgraph_nodes = nx.ego_graph(self.manager.g, start_node_id, radius=depth).nodes()

        # Create the subgraph from the identified nodes
        subgraph_view = self.manager.g.subgraph(subgraph_nodes)

        # Return a copy to prevent modifications affecting the main graph view
        self.logger.debug(
            f"Extracted subgraph for '{term}' (depth {depth}) with {subgraph_view.number_of_nodes()} nodes."
        )
        return subgraph_view.copy()

    @eidosian()
    def get_relationships_by_dimension(
        self,
        dimension: RelationshipDimension = "lexical",
        rel_type: Optional[RelType] = None,
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[Term, Term, RelType, Dict[str, Any]]]:
        """
        Retrieve relationships filtered by dimension and other optional criteria.

        Allows filtering edges based on their assigned dimension (e.g., 'lexical',
        'emotional'), relationship type, and potentially other attributes like
        emotional valence range (if applicable to the dimension).

        Args:
            dimension: The relationship dimension to filter by (default: 'lexical').
            rel_type: Optional specific relationship type to filter by.
            valence_range: Optional tuple (min_valence, max_valence) to filter
                           emotional relationships (applied only if dimension is
                           'emotional' or 'affective').

        Returns:
            List[Tuple[Term, Term, RelType, Dict[str, Any]]]: A list of tuples,
            each containing (source_term, target_term, relationship_type, edge_attributes).
        """
        self.logger.debug(
            f"Querying relationships for dimension '{dimension}' (type: {rel_type}, valence: {valence_range})."
        )
        filtered_relationships: List[Tuple[Term, Term, RelType, Dict[str, Any]]] = []

        for u, v, data in self.manager.g.edges(data=True):
            edge_dimension = data.get("dimension")
            edge_rel_type = data.get("relationship")

            # --- Dimension Filter ---
            if edge_dimension != dimension:
                continue

            # --- Relationship Type Filter ---
            if rel_type is not None and edge_rel_type != rel_type:
                continue

            # --- Valence Filter (Apply only if relevant and specified) ---
            apply_valence_filter = valence_range is not None and dimension in [
                "emotional",
                "affective",
            ]
            if apply_valence_filter:
                # Check valence on both nodes (or edge if stored there)
                # This example assumes valence is primarily a node attribute
                valence_u = self.manager.g.nodes[u].get("valence")
                valence_v = self.manager.g.nodes[v].get("valence")
                # Define logic: e.g., include if *either* node is in range
                node_in_range = False
                min_val, max_val = valence_range
                if valence_u is not None and min_val <= valence_u <= max_val:
                    node_in_range = True
                if valence_v is not None and min_val <= valence_v <= max_val:
                    node_in_range = True

                if not node_in_range:
                    continue  # Skip edge if no node meets valence criteria

            # --- Passed Filters: Retrieve terms and add to results ---
            term_u = self.manager.g.nodes[u].get("term", f"ID:{u}")
            term_v = self.manager.g.nodes[v].get("term", f"ID:{v}")

            # Ensure rel_type is not None before adding
            if edge_rel_type is None:
                self.logger.warning(
                    f"Edge ({u}, {v}) matches dimension '{dimension}' but lacks 'relationship' attribute."
                )
                edge_rel_type = "unknown"  # Assign default

            filtered_relationships.append((term_u, term_v, edge_rel_type, data))

        self.logger.debug(f"Found {len(filtered_relationships)} relationships matching criteria.")
        return filtered_relationships
