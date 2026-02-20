"""
Multidimensional knowledge graph management system for lexical and semantic relationships.

This module provides a comprehensive framework for building, maintaining, and visualizing
knowledge graphs where nodes represent terms and edges represent semantic relationships.
The graph structure supports both 2D and 3D visualizations with rich semantic encoding
of relationship types through colors, weights, and directional properties.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   GraphManager  │────>│  NetworkX Graph │────>│ Visualizations  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
           │                        │                       │
           ▼                        ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Database      │     │  Layout Engines │     │ Export Formats  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Key Components:
    - GraphManager: Core class providing graph operations
    - Custom exceptions for precise error handling
    - Type definitions for structural integrity
    - Layout algorithms for 2D and 3D visualization
    - Semantic analysis capabilities
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx
import numpy as np
from eidosian_core import eidosian

# Import pyvis with a type ignore comment to fix the missing stub issue
from pyvis.network import Network  # type: ignore
from word_forge.database import database_manager as database_manager  # File/Module
from word_forge.database.database_manager import DBManager  # Class/Instance
from word_forge.exceptions import (
    GraphDataError,  # Added missing import
    GraphDimensionError,
    GraphError,
    GraphVisualizationError,
    NodeNotFoundError,
)
from word_forge.graph.graph_config import (
    GraphConfig,
    GraphInfoDict,
    LayoutAlgorithm,
    Position,
    PositionDict,
    WordId,
)
from word_forge.relationships import RELATIONSHIP_TYPES, RelationshipProperties

config = GraphConfig()  # Create instance first
get_export_path = config.get_export_filepath("default_gexf")


class GraphManager:
    """
    Builds and maintains a multidimensional lexical/semantic graph from database records.

    This class creates a networkx graph representation where:
    - Nodes represent words (with word_id as node identifier)
    - Edges represent relationships between words (synonym, antonym, etc.)
    - Node attributes include the term text and multidimensional positioning
    - Edge attributes include the relationship type, weight, and color
    - The graph can be visualized in 2D or 3D space with rich semantic encoding

    Attributes:
        db_manager: Database manager providing access to word data
        g: The NetworkX graph instance containing nodes and edges
        _term_to_id: Mapping from lowercase terms to their IDs for fast lookup
        _dimensions: Number of dimensions for layout (2 or 3)
        _positions: Dictionary mapping node IDs to spatial coordinates
        _relationship_counts: Counter tracking relationship type frequencies
        _emotional_contexts: Dictionary storing defined emotional contexts.

    Usage Examples:
        ```python
        # Initialize with database manager
        db_manager = DBManager()
        graph_manager = GraphManager(db_manager)

        # Build graph from database
        graph_manager.build_graph()

        # Get information about the graph
        info = graph_manager.get_graph_info()
        print(f"Graph has {info['nodes']} nodes and {info['edges']} edges")

        # Find related terms
        related_terms = graph_manager.get_related_terms("algorithm")

        # Create visualization
        graph_manager.visualize()
        ```
    """

    def __init__(self, db_manager: DBManager) -> None:
        """
        Initialize the graph manager with a database connection.

        Sets up the graph structure, term mappings, and dimensional parameters.
        Ensures the database directory exists to prevent file operation errors.

        Args:
            db_manager: Database manager providing access to word data
        """
        self.db_manager = db_manager
        self.g: nx.Graph = nx.Graph()  # Explicitly specify the type for proper type checking
        self._term_to_id: Dict[str, int] = {}
        self._dimensions: int = 2  # Default is 2D for backward compatibility
        self._positions: Dict[int, Tuple[float, ...]] = {}
        self._relationship_counts: Dict[str, int] = {}
        self._emotional_contexts: Dict[str, Dict[str, float]] = {}  # Initialize contexts

        # Ensure the database parent directory exists
        db_path = Path(self.db_manager.db_path)
        if not db_path.exists() and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory for database: {db_path.parent}")

    @property
    def dimensions(self) -> int:
        """
        Get the number of dimensions for the graph layout.

        Returns:
            The current number of dimensions (2 or 3)
        """
        return self._dimensions

    def dimensions(self, dims: int) -> None:
        """
        Set the number of dimensions for the graph layout.

        Updates the dimensionality parameter which affects layout computation
        and visualization. When changed, requires recomputing layout.

        Args:
            dims: Number of dimensions (2 or 3)

        Raises:
            GraphDimensionError: If dimensions are not 2 or 3

        Example:
            ```python
            # Set to 3D for spatial visualization
            graph_manager.dimensions = 3
            graph_manager._compute_layout()  # Recompute layout
            ```
        """
        if dims not in (2, 3):
            raise GraphDimensionError(
                "Graph dimensions must be either 2 or 3",
                ValueError(f"Invalid dimension: {dims}"),
            )
        self._dimensions = dims

    @contextmanager
    def _db_connection(self) -> sqlite3.Connection:
        """
        Create a database connection using the DBManager's path.

        Context manager that provides a database connection with proper
        setup and guaranteed cleanup regardless of exceptions.
        Ensures the parent directory exists before connecting.

        Yields:
            sqlite3.Connection: An active database connection with row factory

        Raises:
            GraphError: If database connection fails

        Example:
            ```python
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
            # Connection automatically closed here
            ```
        """
        conn: Optional[sqlite3.Connection] = None  # Ensure conn is defined before try
        try:
            # Ensure parent directory exists
            db_path = Path(self.db_manager.db_path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory for database: {db_path.parent}")

            conn = sqlite3.connect(self.db_manager.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            error_msg = f"Database connection error in graph manager: {e}"
            raise GraphError(error_msg, e) from e
        finally:
            if conn:
                conn.close()

    @eidosian()
    def build_graph(self) -> None:
        """
        Clear existing in-memory graph and rebuild from database.

        This method:
        1. Clears any existing graph data
        2. Fetches words and relationships from the database
        3. Adds each word as a node with its term as an attribute
        4. Adds edges between words based on their relationships
        5. Computes layout positions in 2D or 3D space

        Raises:
            GraphError: If a database error occurs during fetching

        Example:
            ```python
            # Complete rebuild from database
            graph_manager.build_graph()
            ```
        """
        self.g.clear()
        self._term_to_id.clear()
        self._positions.clear()
        self._relationship_counts = {}

        words, relationships = self._fetch_data()

        # Add nodes and build term->id mapping
        for word_id, term in words:
            if term:  # Ensure term is not None or empty before adding
                self.g.add_node(word_id, term=term, id=word_id)  # Add id attribute here
                self._term_to_id[term.lower()] = word_id
                # self.g.nodes[word_id]["id"] = word_id # Redundant, added in add_node
                # self.g.nodes[word_id]["term"] = term # Redundant, added in add_node

        # Add edges between related words with rich attributes
        for word_id, related_term, rel_type in relationships:
            # Skip if either node doesn't exist
            if word_id not in self.g:  # Use 'in' for faster check
                continue

            # Find ID for related term (skip if not found)
            related_id = self._term_to_id.get(related_term.lower())
            if related_id is None or related_id not in self.g:  # Use 'in'
                continue

            # Get relationship properties
            rel_props = self._get_relationship_properties(rel_type)
            weight = rel_props["weight"]
            color = rel_props["color"]
            bidirectional = rel_props["bidirectional"]

            # Determine relationship dimension
            dimension = "lexical"  # Default dimension
            if any(
                rel_type.startswith(prefix)
                for prefix in [
                    "emotional_",
                    "evokes",
                    "responds_to",
                    "valence_",
                    "arousal_",
                    "meta_emotion",  # Added meta_emotion prefix
                    "intensifies",  # Added intensifies
                    "diminishes",  # Added diminishes
                    "emotional_component",  # Added emotional_component
                ]
            ):
                dimension = "emotional"
            elif any(rel_type.startswith(prefix) for prefix in ["positive_", "negative_", "high_", "low_"]):
                dimension = "affective"

            # Add edge with rich attributes
            source_term_text = self.g.nodes[word_id].get("term", "Unknown")
            target_term_text = self.g.nodes[related_id].get("term", "Unknown")
            self.g.add_edge(
                word_id,
                related_id,
                relationship=rel_type,  # Use 'relationship' consistently
                weight=weight,
                color=color,
                bidirectional=bidirectional,
                dimension=dimension,
                title=f"{rel_type}: {source_term_text} → {target_term_text}",
            )

            # Track relationship counts for statistics
            self._relationship_counts[rel_type] = self._relationship_counts.get(rel_type, 0) + 1

        # Generate layout positions
        self._compute_layout()

    def _get_relationship_properties(self, rel_type: str) -> RelationshipProperties:
        """
        Get the properties for a given relationship type.

        Args:
            rel_type: The relationship type

        Returns:
            Dictionary with weight, color, and bidirectional properties
        """
        # Ensure result is correctly typed
        result: RelationshipProperties = RELATIONSHIP_TYPES.get(rel_type, RELATIONSHIP_TYPES["default"])
        return result

    def _compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """
        Compute node positions for the graph using the specified algorithm.

        Calculates spatial positions for all nodes using the selected layout
        algorithm, with different options available for 2D and 3D visualization.
        Results are stored in both the _positions dictionary and as node attributes.

        Args:
            algorithm: Layout algorithm to use (defaults to config.default_layout)
                Options include: "force_directed", "spectral", "circular"

        Raises:
            GraphError: If layout computation fails

        Example:
            ```python
            # Compute with force-directed layout
            graph_manager._compute_layout("force_directed")

            # Compute with default layout algorithm
            graph_manager._compute_layout()
            ```
        """
        if self.g.number_of_nodes() == 0:
            logging.warning("Attempted to compute layout on an empty graph.")
            return

        try:
            # Choose layout algorithm based on dimensions
            algorithm_to_use: LayoutAlgorithm = algorithm or config.default_layout  # Use config default

            pos: PositionDict = self._calculate_layout_positions(algorithm_to_use)

            # Store positions
            self._positions = pos

            # Add positions as node attributes
            for node_id, position in pos.items():
                if node_id in self.g:  # Ensure node exists before adding attributes
                    if self._dimensions == 3 and len(position) == 3:
                        self.g.nodes[node_id]["x"] = float(position[0])
                        self.g.nodes[node_id]["y"] = float(position[1])
                        self.g.nodes[node_id]["z"] = float(position[2])
                    elif self._dimensions == 2 and len(position) >= 2:
                        self.g.nodes[node_id]["x"] = float(position[0])
                        self.g.nodes[node_id]["y"] = float(position[1])
                    else:
                        logging.warning(
                            f"Node {node_id} has position {position} incompatible with dimension {self._dimensions}"
                        )

        except Exception as e:
            error_msg = f"Failed to compute layout using '{algorithm_to_use}': {e}"
            raise GraphError(error_msg, e) from e

    def _calculate_layout_positions(self, algorithm: LayoutAlgorithm) -> PositionDict:
        """
        Calculate node positions using a specified NetworkX layout algorithm.

        Args:
            algorithm: Layout algorithm name (force_directed, spectral, circular)

        Returns:
            Dictionary mapping node IDs to position tuples

        Raises:
            GraphError: If the layout algorithm fails or is unknown
        """
        layout_func: Optional[Callable] = None
        kwargs: Dict[str, Any] = {"G": self.g, "weight": "weight"}

        if self._dimensions == 3:
            kwargs["dim"] = 3
            kwargs["scale"] = 2.0  # Consistent scaling for 3D
            if algorithm == "force_directed":
                layout_func = nx.spring_layout
            elif algorithm == "spectral":
                # Spectral layout extended to 3D (custom implementation)
                try:
                    pos_2d = nx.spectral_layout(self.g, weight="weight")
                    bc = nx.betweenness_centrality(self.g, weight="weight")
                    # Ensure all nodes from pos_2d are in the graph before accessing centrality
                    return {
                        n: (
                            *pos_2d[n],
                            bc.get(n, 0.0) * kwargs["scale"],
                        )  # Scale Z by scale factor
                        for n in pos_2d
                        if n in self.g
                    }
                except Exception as e:
                    raise GraphError(f"Failed during 3D spectral layout calculation: {e}", e) from e
            else:  # Default to 3D spring layout if algorithm is unknown or not spectral
                logging.warning(
                    f"Unknown or unsupported 3D layout algorithm '{algorithm}', defaulting to 'force_directed'."
                )
                layout_func = nx.spring_layout
        else:  # 2D layout
            kwargs["dim"] = 2
            if algorithm == "force_directed":
                layout_func = nx.spring_layout
            elif algorithm == "spectral":
                layout_func = nx.spectral_layout
            elif algorithm == "circular":
                layout_func = nx.circular_layout
            else:
                logging.warning(f"Unknown 2D layout algorithm '{algorithm}', defaulting to 'force_directed'.")
                layout_func = nx.spring_layout

        if layout_func:
            try:
                # Remove 'scale' for algorithms that don't support it directly in 2D
                if self._dimensions == 2 and algorithm != "force_directed":
                    kwargs.pop("scale", None)
                # Ensure node IDs are correctly typed if necessary (NetworkX usually handles this)
                positions_raw = layout_func(**kwargs)
                # Ensure positions are tuples of floats
                return {node: tuple(float(coord) for coord in pos) for node, pos in positions_raw.items()}
            except Exception as e:
                raise GraphError(f"Layout algorithm '{algorithm}' failed: {e}", e) from e
        else:
            # This case should ideally not be reached due to defaults, but added for safety
            raise GraphError(
                f"Layout function could not be determined for algorithm '{algorithm}' and dimension {self._dimensions}"
            )

    @eidosian()
    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """
        Save the current graph to a GEXF format file.

        Exports the graph to the GEXF format for use with external tools
        like Gephi or for later reloading. Creates parent directories
        if they don't exist.

        Args:
            path: Destination file path (defaults to config value)

        Raises:
            ValueError: If the graph is empty
            GraphError: If writing to the file fails

        Example:
            ```python
            # Save to default path
            graph_manager.save_to_gexf()

            # Save to custom path
            graph_manager.save_to_gexf("data/my_graph.gexf")
            ```
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot save an empty graph")

        gexf_path_str = path or str(get_export_path)
        gexf_path = Path(gexf_path_str)
        try:
            # Ensure directory exists
            gexf_path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_gexf(self.g, str(gexf_path))  # Pass path as string
            logging.info(f"Graph saved successfully to {gexf_path}")
        except Exception as e:
            error_msg = f"Failed to save graph to {gexf_path}: {e}"
            raise GraphError(error_msg, e) from e

    @eidosian()
    def load_from_gexf(self, path: Optional[str] = None) -> None:
        """
        Load a graph from a GEXF format file.

        Imports a previously saved graph or one created by external tools.
        Reconstructs all internal mappings (_term_to_id, _positions) and
        detects dimensionality from the file.

        Args:
            path: Source file path (defaults to config value)

        Raises:
            FileNotFoundError: If the file doesn't exist
            GraphError: If the file format is invalid

        Example:
            ```python
            # Load from default path
            graph_manager.load_from_gexf()

            # Load from custom path
            graph_manager.load_from_gexf("data/my_graph.gexf")
            ```
        """
        gexf_path_str = path or str(get_export_path)
        gexf_path = Path(gexf_path_str)
        if not gexf_path.exists():
            raise FileNotFoundError(f"Graph file not found: {gexf_path}")

        try:
            # Specify node_type=int if node IDs are expected to be integers
            self.g = nx.read_gexf(str(gexf_path), node_type=int)  # Pass path as string

            # Rebuild term_to_id mapping from loaded graph
            self._term_to_id.clear()
            for node_id, attrs in self.g.nodes(data=True):
                # Ensure node_id is int for consistency
                node_id_int = int(node_id)
                if "term" in attrs and isinstance(attrs["term"], str):
                    self._term_to_id[attrs["term"].lower()] = node_id_int

            # Detect dimensionality of loaded graph
            first_node_attrs = next(iter(self.g.nodes(data=True)), (None, {}))[1]
            if "z" in first_node_attrs and first_node_attrs.get("z") is not None:
                self._dimensions = 3
            elif "x" in first_node_attrs and "y" in first_node_attrs:
                self._dimensions = 2
            else:
                self._dimensions = 2  # Default if no position data found
                logging.warning("No position data found in GEXF, defaulting to 2D.")

            # Rebuild positions from node attributes
            self._positions = {}
            for node_id, attrs in self.g.nodes(data=True):
                node_id_int = int(node_id)  # Ensure int key
                try:
                    if self._dimensions == 3 and all(k in attrs for k in ("x", "y", "z")):
                        self._positions[node_id_int] = (
                            float(attrs["x"]),
                            float(attrs["y"]),
                            float(attrs["z"]),
                        )
                    elif self._dimensions == 2 and all(k in attrs for k in ("x", "y")):
                        self._positions[node_id_int] = (
                            float(attrs["x"]),
                            float(attrs["y"]),
                        )
                except (ValueError, TypeError) as pos_err:
                    logging.warning(f"Could not parse position for node {node_id_int}: {pos_err}")

            # Rebuild relationship counts
            self._relationship_counts = {}
            for _, _, data in self.g.edges(data=True):
                # Use 'relationship' key consistent with build_graph
                rel_type = data.get("relationship", "default")
                if isinstance(rel_type, str):  # Ensure rel_type is a string
                    self._relationship_counts[rel_type] = self._relationship_counts.get(rel_type, 0) + 1
            logging.info(f"Graph loaded successfully from {gexf_path}")

        except Exception as e:
            error_msg = f"Failed to load graph from {gexf_path}: {e}"
            raise GraphError(error_msg, e) from e

    @eidosian()
    def update_graph(self) -> int:
        """
        Update existing graph with new words and relationships from the database.

        Instead of clearing the graph, this method:
        1. Identifies words in the database not yet in the graph
        2. Adds them as new nodes
        3. Adds relationships involving the new words or existing words if missing
        4. Updates layout positions preserving existing node positions where possible

        Returns:
            Number of new words added to the graph

        Raises:
            GraphError: If database access fails

        Example:
            ```python
            # Add any new words from the database
            new_words = graph_manager.update_graph()
            print(f"Added {new_words} new terms to the graph")
            ```
        """
        if self.g.number_of_nodes() == 0:
            # If graph is empty, just build it from scratch
            self.build_graph()
            return self.g.number_of_nodes()

        # Get current words in the graph
        current_ids: Set[int] = set(self.g.nodes())  # Assuming node IDs are int

        # Fetch all words and relationships from database
        all_words, all_relationships = self._fetch_data()

        # Find words not yet in the graph
        new_words = [(word_id, term) for word_id, term in all_words if word_id not in current_ids and term]
        new_word_count = len(new_words)

        # Add new words to the graph
        for word_id, term in new_words:
            if term:  # Double check term is valid
                self.g.add_node(word_id, term=term, id=word_id)
                self._term_to_id[term.lower()] = word_id

        # Add relationships (new or existing nodes) if they don't exist yet
        new_edges = 0
        for word_id, related_term, rel_type in all_relationships:
            # Get related id from term
            related_id = self._term_to_id.get(related_term.lower())

            # Check if both nodes exist in the graph now and edge is missing
            if word_id in self.g and related_id is not None and related_id in self.g:
                if not self.g.has_edge(word_id, related_id):
                    # Get relationship attributes
                    rel_props = self._get_relationship_properties(rel_type)
                    dimension = self._determine_dimension(rel_type)  # Helper function
                    source_term_text = self.g.nodes[word_id].get("term", "Unknown")
                    target_term_text = self.g.nodes[related_id].get("term", "Unknown")

                    # Add edge with attributes
                    self.g.add_edge(
                        word_id,
                        related_id,
                        relationship=rel_type,  # Consistent key
                        weight=rel_props["weight"],
                        color=rel_props["color"],
                        bidirectional=rel_props["bidirectional"],
                        dimension=dimension,
                        title=f"{rel_type}: {source_term_text} → {target_term_text}",
                    )
                    new_edges += 1

                    # Update relationship counts
                    self._relationship_counts[rel_type] = self._relationship_counts.get(rel_type, 0) + 1

        # Update layout only if we added new nodes/edges
        if new_word_count > 0 or new_edges > 0:
            self._update_layout_incrementally()
            logging.info(f"Updated graph: Added {new_word_count} nodes and {new_edges} edges.")
        else:
            logging.info("Graph update: No new nodes or edges added.")

        return new_word_count

    def _determine_dimension(self, rel_type: str) -> str:
        """Determine the dimension based on relationship type"""
        if any(
            rel_type.startswith(prefix)
            for prefix in [
                "emotional_",
                "evokes",
                "responds_to",
                "valence_",
                "arousal_",
                "meta_emotion",
                "intensifies",
                "diminishes",
                "emotional_component",
            ]
        ):
            return "emotional"
        elif any(rel_type.startswith(prefix) for prefix in ["positive_", "negative_", "high_", "low_"]):
            return "affective"
        else:
            return "lexical"

    def _update_layout_incrementally(self) -> None:
        """
        Update layout positions incrementally, preserving existing node positions.

        This method:
        1. Keeps positions of existing nodes
        2. Places new nodes based on their connections
        3. Fine-tunes all positions with a few iterations

        Raises:
            GraphError: If layout update fails
        """
        try:
            # Get nodes without positions (new nodes)
            nodes_without_pos: List[int] = [int(n) for n in self.g.nodes() if n not in self._positions]

            if not nodes_without_pos:
                logging.info("Incremental layout update: No new nodes to position.")
                return

            # Create initial positions for new nodes based on dimensions
            self._initialize_new_node_positions(nodes_without_pos)

            # Fine-tune positions with a few iterations of force-directed layout
            # but only move new nodes significantly
            # Use current positions as starting point
            pos_start = {nid: tuple(p) for nid, p in self._positions.items() if nid in self.g}  # Ensure tuples

            # Identify nodes that already had positions
            fixed_nodes = [n for n in self.g.nodes() if n in pos_start and n not in nodes_without_pos]

            # Run spring layout, fixing old nodes, starting from current positions
            pos_updated = nx.spring_layout(
                self.g,
                pos=pos_start,  # Start from existing/initialized positions
                fixed=fixed_nodes if fixed_nodes else None,  # Fix only existing nodes
                weight="weight",
                iterations=50,  # Fewer iterations for incremental update
                dim=self._dimensions,
                scale=2.0 if self._dimensions == 3 else 1.0,  # Use appropriate scale
            )

            # Update positions - convert numpy arrays to tuples to match type annotation
            self._positions = {node: tuple(float(p) for p in position) for node, position in pos_updated.items()}

            # Update node attributes with new positions
            for node_id, position in self._positions.items():
                if node_id in self.g:  # Check node exists
                    if self._dimensions == 3 and len(position) == 3:
                        self.g.nodes[node_id]["x"] = position[0]
                        self.g.nodes[node_id]["y"] = position[1]
                        self.g.nodes[node_id]["z"] = position[2]
                    elif self._dimensions == 2 and len(position) >= 2:
                        self.g.nodes[node_id]["x"] = position[0]
                        self.g.nodes[node_id]["y"] = position[1]

            logging.info(f"Incrementally updated layout for {len(nodes_without_pos)} new nodes.")

        except Exception as e:
            error_msg = f"Failed to update graph layout incrementally: {e}"
            raise GraphError(error_msg, e) from e

    def _initialize_new_node_positions(self, nodes_without_pos: List[int]) -> None:
        """
        Initialize positions for newly added nodes based on graph topology.

        Positions new nodes near their neighbors when possible, or randomly
        when no connected neighbors have positions. Handles both 2D and 3D cases.

        Args:
            nodes_without_pos: List of node IDs that need positioning
        """
        for node in nodes_without_pos:
            if node not in self.g:
                continue  # Skip if node somehow isn't in graph

            # Explicitly type the neighbors list to satisfy type checker
            neighbors: List[int] = list(self.g.neighbors(node))  # Neighbors are node IDs (int)

            # Get positions of neighbors that *already* have positions
            neighbor_positions = [
                self._positions[n]  # Directly access, assuming tuple[float,...]
                for n in neighbors
                if n in self._positions
            ]

            if not neighbor_positions:
                # No positioned neighbors, place randomly
                self._assign_random_position(node)
                continue

            # Calculate average position of neighbors
            avg_pos: Position = self._calculate_average_position(neighbor_positions)

            # Add some randomness to avoid overlap, ensure floats
            noise_scale = 0.1
            if self._dimensions == 3 and len(avg_pos) == 3:
                self._positions[node] = (
                    float(avg_pos[0] + noise_scale * np.random.randn()),
                    float(avg_pos[1] + noise_scale * np.random.randn()),
                    float(avg_pos[2] + noise_scale * np.random.randn()),
                )
            elif self._dimensions == 2 and len(avg_pos) >= 2:
                self._positions[node] = (
                    float(avg_pos[0] + noise_scale * np.random.randn()),
                    float(avg_pos[1] + noise_scale * np.random.randn()),
                )
            else:  # Fallback if avg_pos dimension mismatch
                self._assign_random_position(node)

    def _assign_random_position(self, node: int) -> None:
        """
        Assign a random position to a node based on current dimensions.

        Args:
            node: ID of the node to position
        """
        scale = 1.0  # Scale for random positions
        if self._dimensions == 3:
            self._positions[node] = (
                np.random.uniform(-scale, scale),
                np.random.uniform(-scale, scale),
                np.random.uniform(-scale, scale),
            )
        else:
            self._positions[node] = (
                np.random.uniform(-scale, scale),
                np.random.uniform(-scale, scale),
            )

    def _get_zero_position(self) -> Position:
        """
        Get a zero position tuple based on current dimensions.

        Returns:
            A 2D or 3D tuple of zeros
        """
        return (0.0, 0.0, 0.0) if self._dimensions == 3 else (0.0, 0.0)

    def _calculate_average_position(self, positions: List[Position]) -> Position:
        """
        Calculate the average position from a list of position tuples.

        Args:
            positions: List of position tuples (e.g., [(x1, y1), (x2, y2)])

        Returns:
            Average position as a tuple (e.g., (avg_x, avg_y))
        """
        if not positions:
            return self._get_zero_position()

        num_positions = len(positions)
        # Determine dimension from the first position tuple
        pos_dim = len(positions[0])

        if pos_dim == 3 and self._dimensions == 3:
            avg_x = sum(p[0] for p in positions) / num_positions
            avg_y = sum(p[1] for p in positions) / num_positions
            avg_z = sum(p[2] for p in positions) / num_positions
            return (avg_x, avg_y, avg_z)
        elif pos_dim >= 2:  # Handle 2D or higher-D positions when graph is 2D
            avg_x = sum(p[0] for p in positions) / num_positions
            avg_y = sum(p[1] for p in positions) / num_positions
            return (avg_x, avg_y)
        else:  # Fallback for unexpected position format
            return self._get_zero_position()

    @eidosian()
    def ensure_sample_data(self) -> bool:
        """
        Ensure the database contains sample data if it's empty.

        Checks if the database has any words, and if not, populates it
        with predefined sample data from configuration. Adds both sample
        words and their relationships.

        Returns:
            True if sample data was added, False if database already had data

        Raises:
            GraphError: If adding sample data fails

        Example:
            ```python
            # Ensure there's at least sample data
            if graph_manager.ensure_sample_data():
                print("Added sample data to empty database")
            ```
        """
        # Check if the database has any words
        try:
            words, _ = self._fetch_data()
            if words:
                logging.info("Database already contains data. Skipping sample data insertion.")
                return False
        except GraphError as e:
            # If fetching fails (e.g., tables don't exist), proceed to add data
            logging.warning(f"Could not fetch initial data, attempting to add sample data: {e}")

        # Database is empty or tables missing, add sample data
        logging.info("Database appears empty. Attempting to add sample data.")
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Ensure tables exist (create if not) - Use DBManager's setup
                self.db_manager.setup_database(conn)

                # Add sample words if not present
                inserted_word_ids: Dict[str, int] = {}
                for word_data in config.sample_words:  # Use sample_words from config
                    term = word_data.get("term")
                    definition = word_data.get("definition", "")
                    pos = word_data.get("part_of_speech", "")

                    if term:
                        try:
                            cursor.execute(
                                database_manager.SQL_INSERT_WORD,  # Use standard insert
                                (term, definition, pos),
                            )
                            inserted_id = cursor.lastrowid
                            if inserted_id is not None:
                                inserted_word_ids[term.lower()] = inserted_id
                            else:
                                # Fetch ID if lastrowid is not supported/returned
                                cursor.execute("SELECT id FROM words WHERE term = ?", (term,))
                                row = cursor.fetchone()
                                if row:
                                    inserted_word_ids[term.lower()] = row[0]
                        except sqlite3.IntegrityError:
                            # Word might already exist, fetch its ID
                            cursor.execute("SELECT id FROM words WHERE term = ?", (term,))
                            row = cursor.fetchone()
                            if row:
                                inserted_word_ids[term.lower()] = row[0]
                            logging.warning(f"Sample word '{term}' already exists.")
                        except sqlite3.Error as insert_err:
                            logging.error(f"Error inserting sample word '{term}': {insert_err}")

                # Add sample relationships from config
                for rel_data in config.sample_relationships:  # Use sample_relationships
                    term1 = rel_data.get("term1")
                    term2 = rel_data.get("term2")
                    rel_type = rel_data.get("relationship_type")

                    if term1 and term2 and rel_type:
                        id1 = inserted_word_ids.get(term1.lower())
                        id2 = inserted_word_ids.get(term2.lower())  # Need ID for term2 as well

                        if id1 is not None and id2 is not None:
                            try:
                                # Use standard relationship insert SQL from database_manager
                                cursor.execute(
                                    database_manager.SQL_INSERT_RELATIONSHIP,
                                    (
                                        id1,
                                        term2,
                                        rel_type,
                                    ),  # Insert using word_id, related_term text
                                )
                            except sqlite3.IntegrityError:
                                logging.warning(f"Sample relationship {term1}-{rel_type}-{term2} already exists.")
                            except sqlite3.Error as rel_err:
                                logging.error(
                                    f"Error inserting sample relationship {term1}-{rel_type}-{term2}: {rel_err}"
                                )
                        else:
                            logging.warning(
                                f"Could not find IDs for sample relationship: {term1} ({id1}) - {term2} ({id2})"
                            )

                conn.commit()
                logging.info("Successfully added sample data to the database.")
                return True

        except sqlite3.Error as e:
            error_msg = f"Failed to add sample data: {e}"
            raise GraphError(error_msg, e) from e
        except Exception as e:  # Catch broader exceptions during setup/insert
            error_msg = f"An unexpected error occurred while adding sample data: {e}"
            raise GraphError(error_msg, e) from e

    def _fetch_data(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str, str]]]:
        """
        Fetch words and relationships from the database.

        Returns:
            Tuple containing (word_tuples, relationship_tuples) where:
            - word_tuples: List of (word_id: int, term: str)
            - relationship_tuples: List of (word_id: int, related_term: str, relationship_type: str)

        Raises:
            GraphError: If database access fails or tables are missing.
        """
        words: List[Tuple[int, str]] = []
        relationships: List[Tuple[int, str, str]] = []
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Check if words table exists before querying
                cursor.execute(database_manager.SQL_CHECK_WORDS_TABLE)
                if not cursor.fetchone():
                    raise GraphError("Database table 'words' not found.")

                # Get all words - use id instead of word_id to match actual schema
                cursor.execute("SELECT id, term FROM words")
                # Fetchall returns list of Row objects, convert to tuples
                words_raw = cursor.fetchall()
                words = [
                    (row["id"], row["term"]) for row in words_raw if row["id"] is not None and row["term"] is not None
                ]

                # Check if relationships table exists
                cursor.execute(database_manager.SQL_CHECK_RELATIONSHIPS_TABLE)
                if not cursor.fetchone():
                    logging.warning("Database table 'relationships' not found. Graph will have no edges.")
                    return (
                        words,
                        [],
                    )  # Return words without relationships if table missing

                # Get relationships - ensure column names match the actual schema
                cursor.execute("""
                    SELECT word_id, related_term, relationship_type
                    FROM relationships
                """)
                relationships_raw = cursor.fetchall()
                relationships = [
                    (row["word_id"], row["related_term"], row["relationship_type"])
                    for row in relationships_raw
                    if row["word_id"] is not None
                    and row["related_term"] is not None
                    and row["relationship_type"] is not None
                ]

                return words, relationships
        except sqlite3.Error as db_err:
            raise GraphError(f"Failed to fetch graph data due to database error: {db_err}", db_err) from db_err
        except Exception as e:
            # Ensure proper error propagation with cause parameter
            raise GraphError(f"Failed to fetch graph data: {e}", e) from e

    @eidosian()
    def get_related_terms(self, term: str, rel_type: Optional[str] = None) -> List[str]:
        """
        Find terms related to the given term, optionally filtered by relationship type.

        Traverses the graph to find connected terms, with optional filtering
        by relationship type (e.g., only synonyms).

        Args:
            term: The term to find relationships for
            rel_type: Optional relationship type filter (e.g., 'synonym', 'antonym')

        Returns:
            List of related terms

        Raises:
            NodeNotFoundError: If the term is not found in the graph

        Example:
            ```python
            # Get all related terms
            all_related = graph_manager.get_related_terms("algorithm")

            # Get only synonyms
            synonyms = graph_manager.get_related_terms("algorithm", rel_type="synonym")
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term not found in graph: {term}")

        term_id = self._term_to_id[term_lower]
        related_terms = []

        if term_id not in self.g:
            # This case should ideally not happen if _term_to_id is consistent with g
            raise NodeNotFoundError(f"Term ID {term_id} for '{term}' not found in graph nodes.")

        for neighbor_id in self.g.neighbors(term_id):
            edge_data = self.g.get_edge_data(term_id, neighbor_id)
            if edge_data:  # Ensure edge data exists
                # Use 'relationship' key consistent with build_graph
                neighbor_rel_type = edge_data.get("relationship")

                # Filter by relationship type if specified
                if rel_type and neighbor_rel_type != rel_type:
                    continue

                # Ensure neighbor_id exists in graph nodes before accessing attributes
                if neighbor_id in self.g:
                    neighbor_term = self.g.nodes[neighbor_id].get("term")
                    if neighbor_term:
                        related_terms.append(neighbor_term)
                else:
                    logging.warning(f"Neighbor ID {neighbor_id} found in edges but not in graph nodes.")

        return related_terms

    @eidosian()
    def get_node_count(self) -> int:
        """
        Get the number of word nodes in the graph.

        Returns:
            Integer count of nodes

        Example:
            ```python
            count = graph_manager.get_node_count()
            print(f"Graph contains {count} terms")
            ```
        """
        return self.g.number_of_nodes()

    @eidosian()
    def get_edge_count(self) -> int:
        """
        Get the number of relationship edges in the graph.

        Returns:
            Integer count of edges

        Example:
            ```python
            count = graph_manager.get_edge_count()
            print(f"Graph contains {count} relationships")
            ```
        """
        return self.g.number_of_edges()

    @eidosian()
    def get_term_by_id(self, word_id: int) -> Optional[str]:  # Ensure WordId is int
        """
        Retrieve the term associated with a given word ID.

        Args:
            word_id: The ID of the word to look up

        Returns:
            The term string or None if not found

        Example:
            ```python
            term = graph_manager.get_term_by_id(42)
            if term:
                print(f"Word ID 42 is the term '{term}'")
            ```
        """
        if word_id in self.g:
            # Ensure the attribute exists and is a string
            term = self.g.nodes[word_id].get("term")
            return str(term) if term is not None else None
        return None

    @eidosian()
    def get_graph_info(self) -> GraphInfoDict:
        """
        Get detailed information about the graph structure.

        Collects statistics and sample data to provide an overview
        of the graph's content and structure.

        Returns:
            Dictionary with graph statistics and sample data

        Raises:
            GraphDataError: If the graph structure is invalid or data access fails.

        Example:
            ```python
            info = graph_manager.get_graph_info()
            print(f"Graph has {info['nodes']} nodes and {info['edges']} edges")
            print(f"Relationship types: {info['relationship_types']}")
            ```
        """
        if not isinstance(self.g, nx.Graph):
            raise GraphDataError("Graph object 'g' is not a valid NetworkX graph.")

        try:
            num_nodes = self.g.number_of_nodes()
            num_edges = self.g.number_of_edges()

            # Sample nodes (up to 5)
            sample_nodes = []
            node_iterator = iter(self.g.nodes(data=True))
            for _ in range(min(5, num_nodes)):
                try:
                    node_id, attrs = next(node_iterator)
                    term = attrs.get("term", "Unknown")
                    sample_nodes.append({"id": int(node_id), "term": str(term)})  # Ensure types
                except StopIteration:
                    break

            # Sample relationships (up to 5)
            sample_relationships = []
            edge_iterator = iter(self.g.edges(data=True))
            for _ in range(min(5, num_edges)):
                try:
                    n1, n2, attrs = next(edge_iterator)
                    # Ensure nodes exist before accessing attributes
                    term1 = self.g.nodes[n1].get("term", "Unknown") if n1 in self.g else "Unknown"
                    term2 = self.g.nodes[n2].get("term", "Unknown") if n2 in self.g else "Unknown"
                    # Use 'relationship' key consistent with build_graph
                    rel_type = attrs.get("relationship", "related")
                    sample_relationships.append(
                        {
                            "source": str(term1),
                            "target": str(term2),
                            "type": str(rel_type),
                        }  # Ensure types
                    )
                except StopIteration:
                    break

            # Get list of relationship types used in the graph
            relationship_types = list(self._relationship_counts.keys())

            # Construct the dictionary with explicit types where possible
            info: GraphInfoDict = {
                "nodes": num_nodes,
                "edges": num_edges,
                "dimensions": self._dimensions,
                "sample_nodes": sample_nodes,  # Type checked above
                "sample_relationships": sample_relationships,  # Type checked above
                "relationship_types": relationship_types,  # List[str]
            }
            return info
        except Exception as e:
            error_msg = f"Error generating graph information: {e}"
            raise GraphDataError(error_msg, e) from e

    @eidosian()
    def display_graph_summary(self) -> None:
        """
        Display a summary of the graph structure to the console.

        Prints key statistics and sample nodes/relationships. Provides a
        human-readable overview of the graph contents.

        Example:
            ```python
            graph_manager.display_graph_summary()
            ```
        """
        try:
            info = self.get_graph_info()

            print("\n--- Graph Summary ---")
            print(f"Nodes: {info['nodes']}")
            print(f"Edges: {info['edges']}")
            print(f"Dimensions: {info['dimensions']}D")

            if info["relationship_types"]:
                print("\nRelationship types:")
                # Sort by count descending for clarity
                sorted_counts = sorted(
                    self._relationship_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for rel_type, count in sorted_counts:
                    props = self._get_relationship_properties(rel_type)
                    color = props.get("color", "#888888")
                    print(f"  - {rel_type}: {count} instances (color: {color})")
            else:
                print("\nRelationship types: None found")

            if info["sample_nodes"]:
                print("\nSample nodes (up to 5):")
                for node in info["sample_nodes"]:
                    print(f"  - Node {node['id']}: {node['term']}")
            else:
                print("\nSample nodes: None")

            if info["sample_relationships"]:
                print("\nSample relationships (up to 5):")
                for rel in info["sample_relationships"]:
                    print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']}")
            else:
                print("\nSample relationships: None")

            print("---------------------\n")

        except GraphDataError as e:
            print(f"Could not display graph summary: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while displaying graph summary: {e}")

    @eidosian()
    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,
        dimensions_filter: Optional[List[str]] = None,  # Renamed for clarity
    ) -> None:
        """
        Generate an interactive HTML visualization of the graph using pyvis.

        Creates an interactive visualization, allowing filtering by relationship
        dimension and choosing between 2D or 3D rendering.

        Args:
            output_path: Path to save the HTML file (defaults to config value).
            height: Height of the visualization canvas (e.g., "800px"). Defaults to config.
            width: Width of the visualization canvas (e.g., "100%"). Defaults to config.
            use_3d: Force 3D rendering if True, 2D if False. Defaults to graph's dimension setting.
            dimensions_filter: List of relationship dimensions to include (e.g., ["lexical", "emotional"]).
                               Defaults to including all dimensions present in the graph.

        Raises:
            GraphVisualizationError: If visualization generation fails.
            ValueError: If the graph is empty.

        Example:
            ```python
            # Default visualization (2D or 3D based on graph setting)
            graph_manager.visualize()

            # Force 3D visualization
            graph_manager.visualize(use_3d=True)

            # Only show emotional relationships in 2D
            graph_manager.visualize(dimensions_filter=["emotional"], use_3d=False)

            # Show lexical and affective but not emotional
            graph_manager.visualize(dimensions_filter=["lexical", "affective"])
            ```
        """
        if self.g.number_of_nodes() == 0:
            raise ValueError("Cannot visualize an empty graph")

        # Determine rendering dimension
        render_3d = use_3d if use_3d is not None else (self._dimensions == 3)

        # Create pyvis network
        net = Network(
            height=height or f"{config.vis_height}px",
            width=width or f"{config.vis_width}px",
            directed=True,  # Assume directed for showing relationship directionality
            notebook=False,  # Assuming not running in a Jupyter notebook
            cdn_resources="remote",  # Use remote CDN for JS/CSS
        )

        # Filter graph by dimensions if specified
        target_graph = self.g
        if dimensions_filter:
            # Create a subgraph containing only edges matching the filter
            edges_to_include = [
                (u, v)
                for u, v, data in self.g.edges(data=True)
                if data.get("dimension", "lexical") in dimensions_filter
            ]
            # edge_subgraph includes nodes connected by these edges
            target_graph = self.g.edge_subgraph(edges_to_include).copy()
            if target_graph.number_of_nodes() == 0:
                logging.warning(
                    f"Filtering by dimensions {dimensions_filter} resulted in an empty graph. Nothing to visualize."
                )
                return

        # Configure visualization options (physics, appearance)
        self._configure_visualization_options(net, render_3d)

        # Add nodes to the pyvis network
        self._add_nodes_to_visualization(net, target_graph, render_3d)

        # Add edges to the pyvis network
        self._add_edges_to_visualization(net, target_graph)

        # Determine output path and ensure directory exists
        viz_path_str = output_path or str(config.get_visualization_path() / "graph.html")
        viz_path = Path(viz_path_str)
        viz_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the visualization graph
        try:
            net.save_graph(str(viz_path))  # Pass path as string
            logging.info(f"Visualization saved successfully to: {viz_path}")
            print(f"Visualization saved to: {viz_path}")  # Also print for user feedback
        except Exception as e:
            error_msg = f"Failed to save visualization to {viz_path}: {e}"
            raise GraphVisualizationError(error_msg, e) from e

    def _configure_visualization_options(self, net: Network, use_3d: bool) -> None:
        """
        Configure pyvis visualization options for the network.

        Sets up physics, node styling, edge styling, and interaction options
        based on whether 3D visualization is enabled. Uses configuration values.

        Args:
            net: The pyvis Network instance to configure.
            use_3d: Whether to enable 3D visualization options.
        """
        # Basic options applicable to both 2D and 3D
        options = f"""
        var options = {{
          "nodes": {{
            "shape": "dot",
            "scaling": {{
              "min": {config.min_node_size},
              "max": {config.max_node_size}
            }},
            "font": {{
              "size": 12,
              "face": "Tahoma"
            }},
             "borderWidth": 1,
             "color": {{
                 "border": "{config.node_border_color}",
                 "background": "{config.node_color}",
                 "highlight": {{
                     "border": "{config.node_highlight_border}",
                     "background": "{config.node_highlight_color}"
                 }},
                 "hover": {{
                     "border": "{config.node_hover_border}",
                     "background": "{config.node_hover_color}"
                 }}
             }}
          }},
          "edges": {{
            "smooth": {{
              "type": "continuous",
              "forceDirection": "none",
              "roundness": 0.5
            }},
            "arrows": {{
              "to": {{ "enabled": true, "scaleFactor": 0.5 }}
            }},
            "font": {{
              "size": 10,
              "align": "top"
            }},
            "scaling": {{
                 "min": {config.min_edge_width},
                 "max": {config.max_edge_width}
            }},
            "color": {{
                 "inherit": false, // Use edge-specific colors
                 "highlight": "{config.edge_highlight_color}",
                 "hover": "{config.edge_hover_color}",
                 "opacity": 0.8
            }}
          }},
          "interaction": {{
            "navigationButtons": true,
            "keyboard": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": true,
            "hover": true
          }},
          "physics": {{
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {{
              "gravitationalConstant": -8000,
              "centralGravity": 0.1,
              "springLength": 150,
              "springConstant": 0.02,
              "damping": 0.09,
              "avoidOverlap": 0.1
            }},
            "stabilization": {{
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 50,
              "onlyDynamicEdges": false,
              "fit": true
            }}
          }}
        }};
        """
        # Add 3D specific options if needed (pyvis handles 3D implicitly if z coords are present)
        # No specific 3D options needed for pyvis beyond providing z coordinates

        net.set_options(options)

    def _add_nodes_to_visualization(self, net: Network, graph_to_render: nx.Graph, use_3d: bool) -> None:
        """
        Add nodes from the target graph to the pyvis visualization network.

        Configures each node with appropriate attributes including:
        - Label (term text)
        - Size (based on connectivity in the rendered graph)
        - Position (from layout calculations, scaled for pyvis)
        - Title (hover information)

        Args:
            net: The pyvis Network instance.
            graph_to_render: The NetworkX graph (potentially filtered) to take nodes from.
            use_3d: Whether to include 3D coordinates.
        """
        if graph_to_render.number_of_nodes() == 0:
            return

        # Calculate max degree in the potentially filtered graph for scaling
        degrees = dict(graph_to_render.degree())
        max_degree = max(degrees.values()) if degrees else 1
        max_degree = max(max_degree, 1)  # Avoid division by zero

        for node_id, node_attrs in graph_to_render.nodes(data=True):
            term = node_attrs.get("term", f"Node {node_id}")
            label = term  # Use term as label
            degree = degrees.get(node_id, 0)
            title = f"Term: {term}<br>ID: {node_id}<br>Connections: {degree}"

            # Size node based on degree (connectivity) within the rendered graph
            size = config.min_node_size + (degree * (config.max_node_size - config.min_node_size) / max_degree)

            # Position in 3D or 2D space if available from the original layout
            pos = self._positions.get(node_id)
            x, y, z = None, None, None
            pos_scale = 200  # Scale factor for pyvis coordinates

            if pos:
                try:
                    if use_3d and self._dimensions == 3 and len(pos) == 3:
                        x = float(pos[0]) * pos_scale
                        y = float(pos[1]) * pos_scale
                        z = float(pos[2]) * pos_scale
                    elif len(pos) >= 2:  # Use 2D position if available or if not rendering 3D
                        x = float(pos[0]) * pos_scale
                        y = float(pos[1]) * pos_scale
                        z = 0.0  # Set z to 0 for 2D rendering in pyvis
                except (ValueError, TypeError, IndexError):
                    logging.warning(f"Invalid position format for node {node_id}: {pos}. Using default.")
                    x, y, z = None, None, None  # Fallback if conversion fails

            # Add node to pyvis network
            net.add_node(
                node_id,
                label=label,
                title=title,
                size=size,
                x=x,  # pyvis handles None positions
                y=y,
                z=z if use_3d else None,  # Only provide z if rendering 3D
                # Color is set globally in options, but could be overridden here
                # color={"border": "#023047", "background": "#219ebc"},
            )

    def _add_edges_to_visualization(self, net: Network, graph_to_render: nx.Graph) -> None:
        """
        Add edges from the target graph to the pyvis network visualization.

        Configures edge appearance based on relationship type and dimension,
        applying appropriate colors, weights, and styles from configuration.

        Args:
            net: The pyvis Network instance to add edges to.
            graph_to_render: The NetworkX graph (potentially filtered) to take edges from.
        """
        if graph_to_render.number_of_edges() == 0:
            return

        relationship_legend = {}  # Track types for potential legend generation

        for source, target, data in graph_to_render.edges(data=True):
            # Get relationship properties from edge data
            rel_type = data.get("relationship", "related")  # Consistent key
            weight = data.get("weight", 1.0)
            color = data.get("color", config.default_edge_color)  # Use config default
            bidirectional = data.get("bidirectional", True)
            dimension = data.get("dimension", "lexical")
            title = data.get("title", f"{rel_type}")  # Use pre-generated title or just type

            # Set width based on weight (scaled according to config)
            # Ensure weight is float for calculation
            try:
                edge_weight_float = float(weight)
            except (ValueError, TypeError):
                edge_weight_float = 1.0  # Default weight if conversion fails

            width = config.min_edge_width + (edge_weight_float * (config.max_edge_width - config.min_edge_width))
            # Clamp width to min/max bounds
            width = max(config.min_edge_width, min(width, config.max_edge_width))

            # Set edge style based on dimension
            dashes = False  # Default solid line
            if dimension == "emotional":
                dashes = [5, 5]  # Dashed line for emotional
            elif dimension == "affective":
                dashes = [2, 3]  # Dotted line for affective

            # Only show arrows for directional relationships
            show_arrow = not bidirectional

            # Add edge to pyvis visualization
            net.add_edge(
                source,
                target,
                width=width,
                color=color,
                title=title,  # Hover text
                arrows={"to": {"enabled": show_arrow, "scaleFactor": 0.5}},
                dashes=dashes,
                # smooth={"type": "continuous"} # Set globally in options
            )

            # Track relationship type for potential legend
            if rel_type not in relationship_legend:
                relationship_legend[rel_type] = {
                    "color": color,
                    "dashes": dashes,
                    "dimension": dimension,
                }

        # Store legend data on the network object if needed later (e.g., for custom HTML)
        # setattr(net, 'relationship_legend', relationship_legend) # Optional

    @eidosian()
    def visualize_3d(self, output_path: Optional[str] = None) -> None:
        """
        Generate a 3D interactive visualization of the graph.

        This is a convenience method that ensures 3D layout and visualization.
        It temporarily sets dimensions to 3 if not already, recomputes the layout
        if needed, and then calls the standard visualization method forcing 3D.

        Args:
            output_path: Path where the HTML visualization will be saved (defaults to config).

        Raises:
            GraphVisualizationError: If visualization creation fails.
            GraphError: If layout computation fails.
        """
        original_dimensions = self._dimensions
        needs_recompute = False
        if self._dimensions != 3:
            logging.info("Temporarily setting dimensions to 3 for 3D visualization.")
            self._dimensions = 3
            needs_recompute = True  # Layout needs update for 3D

        # Check if positions exist and match the target dimension
        if not self._positions or any(len(p) != 3 for p in self._positions.values()):
            needs_recompute = True

        if needs_recompute:
            logging.info("Recomputing layout for 3D visualization.")
            try:
                self._compute_layout()  # Recompute layout in 3D
            except GraphError as e:
                # Restore original dimensions before raising
                self._dimensions = original_dimensions
                raise GraphError(f"Failed to compute 3D layout for visualization: {e}", e) from e

        try:
            # Call the main visualization method with 3D flag forced
            self.visualize(output_path=output_path, use_3d=True)
        except (GraphVisualizationError, ValueError) as e:
            # Log or handle visualization specific errors
            logging.error(f"Failed to generate 3D visualization: {e}")
            # Restore dimensions even if visualization fails
            if original_dimensions != self._dimensions:
                self._dimensions = original_dimensions
            raise  # Re-raise the caught exception
        finally:
            # Restore original dimensions if changed
            if original_dimensions != self._dimensions:
                logging.info(f"Restoring graph dimensions to {original_dimensions}D.")
                self._dimensions = original_dimensions
                # Optionally recompute layout back to original dimension if needed elsewhere
                # self._compute_layout()

    @eidosian()
    def get_subgraph(self, term: str, depth: int = 1) -> nx.Graph:
        """
        Extract a subgraph centered on the given term with specified radius (depth).

        Creates a new graph containing the specified term and all related terms
        within the given number of hops (depth). Useful for focused analysis or
        visualization of a term's local neighborhood. Includes all attributes.

        Args:
            term: The central term for the subgraph.
            depth: How many hops (edge traversals) to include from the center term.
                   depth=0 includes only the center node.
                   depth=1 includes the center node and its immediate neighbors.

        Returns:
            NetworkX graph object containing the subgraph.

        Raises:
            NodeNotFoundError: If the term is not found in the graph.

        Example:
            ```python
            # Get immediate neighborhood of "algorithm"
            local_graph = graph_manager.get_subgraph("algorithm", depth=1)

            # Get extended neighborhood (2 hops)
            extended_graph = graph_manager.get_subgraph("algorithm", depth=2)
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term '{term}' not found in graph.")

        center_id = self._term_to_id[term_lower]

        if center_id not in self.g:
            raise NodeNotFoundError(f"Center node ID {center_id} for term '{term}' not found in graph nodes.")

        # Use NetworkX's ego_graph for efficient subgraph extraction by radius
        # It includes the center node and all nodes within the specified radius (depth)
        # and the edges between them.
        try:
            # ego_graph radius corresponds to depth (number of hops)
            subgraph_nodes = nx.single_source_shortest_path_length(self.g, center_id, cutoff=depth).keys()
            subgraph = self.g.subgraph(subgraph_nodes).copy()
            return subgraph
        except nx.NetworkXError as e:
            # Handle potential NetworkX errors during subgraph creation
            raise GraphError(f"Failed to extract subgraph for '{term}' with depth {depth}: {e}", e) from e

    @eidosian()
    def export_subgraph(self, term: str, depth: int = 1, output_path: Optional[str] = None) -> str:
        """
        Export a subgraph centered on a specific term to a GEXF file.

        Extracts the subgraph using `get_subgraph` and saves it to the specified
        or default GEXF file path.

        Args:
            term: The central term for the subgraph.
            depth: How many hops to include from the center term (default: 1).
            output_path: Destination file path for the GEXF file. Defaults to
                         a path derived from the term and depth in the config's export directory.

        Returns:
            str: The absolute path to the exported GEXF file.

        Raises:
            NodeNotFoundError: If the term is not found in the graph.
            GraphError: If writing the subgraph file fails.
        """
        try:
            subgraph = self.get_subgraph(term, depth)
        except NodeNotFoundError:
            raise  # Re-raise NodeNotFoundError directly
        except GraphError as e:
            raise GraphError(f"Failed to get subgraph before exporting: {e}", e) from e

        # Determine export path
        if output_path:
            export_path_str = output_path
        else:
            # Create a default filename based on term and depth
            safe_term = "".join(c if c.isalnum() else "_" for c in term)
            default_filename = f"subgraph_{safe_term}_depth{depth}.gexf"
            # Use config's method to get the full path in the export directory
            export_path_str = str(config.get_export_filepath(default_filename))

        export_path = Path(export_path_str)

        try:
            # Ensure parent directory exists
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the subgraph to GEXF
            nx.write_gexf(subgraph, str(export_path))  # Pass path as string
            logging.info(f"Subgraph for '{term}' (depth {depth}) exported to {export_path}")
            return str(export_path.resolve())  # Return absolute path
        except Exception as e:
            error_msg = f"Failed to export subgraph to {export_path}: {e}"
            raise GraphError(error_msg, e) from e

    @eidosian()
    def analyze_semantic_clusters(
        self,
        min_community_size: int = 3,
        weight_emotional: float = 1.0,
        emotion_only: bool = False,
        resolution: float = 1.0,  # Added resolution parameter for Louvain
    ) -> Dict[int, List[Dict[str, Union[str, float, List[str], bool, None]]]]:  # Allow None for valence/arousal
        """
        Identify semantic and emotional clusters (communities) in the graph using Louvain.

        Uses the Louvain community detection algorithm to find clusters of related terms.
        Allows weighting or filtering by emotional relationships and incorporates
        emotional metadata (valence, arousal) and meta-emotional connections into the analysis.

        Args:
            min_community_size: Minimum number of terms to be considered a community (default: 3).
            weight_emotional: Weight multiplier for emotional relationships (default: 1.0).
                              Values > 1 increase the influence of emotional ties.
            emotion_only: If True, only considers emotional dimension relationships for clustering (default: False).
            resolution: Resolution parameter for the Louvain algorithm. Higher values lead to more communities. (default: 1.0).

        Returns:
            Dictionary mapping community IDs (int) to lists of term data dictionaries.
            Each term dictionary includes:
            - term (str): The term text.
            - valence (float | None): Emotional valence if available (-1.0 to 1.0).
            - arousal (float | None): Emotional arousal if available (0.0 to 1.0).
            - central (bool): Whether this is a central term in the cluster (based on betweenness centrality).
            - related_dimensions (List[str]): List of relationship dimensions connected to this term within the graph.
            - meta_emotions (List[str]): List of terms linked via meta-emotional relationships (e.g., 'evokes').

        Raises:
            GraphError: If community detection fails.
            ImportError: If the required community detection library (`python-louvain` or compatible NetworkX) is not installed.

        Example:
            ```python
            # Find semantic clusters (default settings)
            clusters = graph_manager.analyze_semantic_clusters()

            # Find emotion-weighted clusters with higher resolution
            emotional_clusters = graph_manager.analyze_semantic_clusters(
                weight_emotional=2.0, resolution=1.5
            )

            # Find pure emotional clusters
            pure_emotional = graph_manager.analyze_semantic_clusters(emotion_only=True)
            ```
        """
        if self.g is None or self.g.number_of_nodes() < min_community_size:
            logging.warning("Graph too small for meaningful cluster analysis.")
            return {}

        try:
            # --- Community Detection Function Selection ---
            PartitionFunction = Callable[[nx.Graph, Optional[str], float], Dict[Any, int]]
            best_partition_func: Optional[PartitionFunction] = None

            try:
                # Try python-louvain package first
                import community as community_louvain  # type: ignore

                # Wrapper to match expected signature (graph, weight_key, resolution)
                @eidosian()
                def louvain_wrapper(graph, weight="weight", resolution=1.0):
                    return community_louvain.best_partition(graph, weight=weight, resolution=resolution)

                best_partition_func = louvain_wrapper
                logging.debug("Using 'python-louvain' for community detection.")
            except (ImportError, AttributeError):
                logging.debug("'python-louvain' not found or incompatible.")
                # Fallback to NetworkX's implementation if available
                try:
                    if hasattr(nx.algorithms.community, "louvain_communities"):
                        # Wrapper for NetworkX's function to return partition dict
                        @eidosian()
                        def nx_louvain_wrapper(graph, weight="weight", resolution=1.0):
                            communities_list = list(
                                nx.algorithms.community.louvain_communities(graph, weight=weight, resolution=resolution)
                            )
                            partition_dict = {}
                            for i, community_set in enumerate(communities_list):
                                for node in community_set:
                                    partition_dict[node] = i
                            return partition_dict

                        best_partition_func = nx_louvain_wrapper
                        logging.debug("Using 'networkx.algorithms.community.louvain_communities'.")
                    else:
                        raise AttributeError("NetworkX version lacks louvain_communities function.")
                except (ImportError, AttributeError) as nx_err:
                    logging.error(f"NetworkX community detection error: {nx_err}")
                    raise ImportError(
                        "Community detection requires 'python-louvain' package or NetworkX 2.7+. "
                        "Install with: pip install python-louvain"
                    ) from nx_err

            if best_partition_func is None:
                # Should not happen if import logic is correct, but safety check
                raise GraphError("Could not find a suitable Louvain community detection function.")

            # --- Graph Preparation for Clustering ---
            # Create a graph copy to modify weights without affecting the original
            graph_for_clustering = cast(nx.Graph, self.g.copy())

            # Apply emotional weighting or filtering
            edges_to_remove = []
            for u, v, data in graph_for_clustering.edges(data=True):
                dimension = data.get("dimension", "lexical")
                current_weight = float(data.get("weight", 1.0))  # Ensure float

                if emotion_only:
                    if dimension != "emotional":
                        edges_to_remove.append((u, v))
                    else:  # Keep emotional edges, apply weight multiplier if needed
                        graph_for_clustering[u][v]["weight"] = current_weight * weight_emotional
                elif weight_emotional != 1.0 and dimension == "emotional":
                    # Apply weight multiplier only to emotional edges
                    graph_for_clustering[u][v]["weight"] = current_weight * weight_emotional
                else:
                    # Ensure weight attribute is float even if not modified
                    graph_for_clustering[u][v]["weight"] = current_weight

            graph_for_clustering.remove_edges_from(edges_to_remove)

            # Check if graph became empty after filtering
            if graph_for_clustering.number_of_nodes() == 0 or graph_for_clustering.number_of_edges() == 0:
                logging.warning("Graph became empty after applying emotion filtering. No clusters found.")
                return {}

            # --- Community Detection Execution ---
            partition = best_partition_func(graph_for_clustering, weight="weight", resolution=resolution)

            # --- Post-processing and Data Collection ---
            communities_raw: Dict[int, List[int]] = {}  # Group nodes by community ID
            for node_id, community_id in partition.items():
                if community_id not in communities_raw:
                    communities_raw[community_id] = []
                communities_raw[community_id].append(node_id)

            # Calculate betweenness centrality on the graph used for clustering
            try:
                centrality = nx.betweenness_centrality(graph_for_clustering, weight="weight", normalized=True)
                # Use median centrality as threshold for 'central' status
                median_centrality = np.median(list(centrality.values())) if centrality else 0.0
            except Exception as cent_err:
                logging.warning(
                    f"Could not calculate centrality for cluster analysis: {cent_err}. 'central' attribute will be False."
                )
                centrality = {}
                median_centrality = 0.0

            # Format results, gathering required data for each term
            communities_result: Dict[int, List[Dict[str, Union[str, float, List[str], bool, None]]]] = {}
            for community_id, node_ids in communities_raw.items():
                if len(node_ids) < min_community_size:
                    continue  # Skip small communities

                community_terms = []
                for node_id in node_ids:
                    if node_id not in self.g:
                        continue  # Ensure node exists in original graph

                    term_data = self.g.nodes[node_id]
                    term = term_data.get("term", f"Unknown_{node_id}")

                    # Get related dimensions and meta-emotions from original graph
                    related_dimensions: Set[str] = set()
                    meta_emotions: List[str] = []
                    for neighbor in self.g.neighbors(node_id):
                        if neighbor in self.g:  # Check neighbor exists
                            edge_data = self.g.get_edge_data(node_id, neighbor)
                            if edge_data:
                                dim = edge_data.get("dimension", "lexical")
                                related_dimensions.add(dim)
                                rel_type = edge_data.get("relationship", "")
                                # Check for meta-emotional relationship types
                                if rel_type in [
                                    "meta_emotion",
                                    "evokes",
                                    "emotional_component",
                                ]:
                                    neighbor_term = self.g.nodes[neighbor].get("term")
                                    if neighbor_term:
                                        meta_emotions.append(str(neighbor_term))

                    # Get valence/arousal (allow None)
                    valence = term_data.get("valence")
                    arousal = term_data.get("arousal")

                    # Determine centrality
                    is_central = centrality.get(node_id, 0.0) > median_centrality

                    community_terms.append(
                        {
                            "term": str(term),
                            "valence": float(valence) if valence is not None else None,
                            "arousal": float(arousal) if arousal is not None else None,
                            "central": is_central,
                            "related_dimensions": sorted(list(related_dimensions)),  # Sort for consistency
                            "meta_emotions": sorted(list(set(meta_emotions))),  # Unique and sorted
                        }
                    )

                if community_terms:  # Only add if list is not empty
                    communities_result[community_id] = community_terms

            return communities_result

        except ImportError as e:
            # Specific handling for missing libraries
            logging.error(f"ImportError during cluster analysis: {e}")
            raise  # Re-raise to signal dependency issue
        except Exception as e:
            error_msg = f"Failed to analyze semantic clusters: {e}"
            logging.error(error_msg, exc_info=True)
            raise GraphError(error_msg, e) from e

    @eidosian()
    def get_relationships_by_dimension(
        self,
        dimension: str = "lexical",
        rel_type: Optional[str] = None,
        include_meta: bool = False,  # Corrected type hint and default
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Get all relationships matching specific dimension and filters.

        Retrieves relationships belonging to the specified dimension (lexical,
        emotional, affective) with optional filtering by relationship type,
        inclusion of meta-emotional links, and emotional valence range.

        Args:
            dimension: Relationship dimension ("lexical", "emotional", "affective").
            rel_type: Optional specific relationship type (e.g., "synonym").
            include_meta: If True and dimension is "lexical", also include edges
                          where the *edge* dimension is "emotional" but links back
                          to the lexical term (representing meta-connections). Default: False.
            valence_range: Optional tuple (min_valence, max_valence) to filter
                          emotional relationships by the source node's valence.

        Returns:
            List of tuples: (source_term, target_term, relationship_type, attributes).
            Attributes dict contains edge data like weight, color, etc.

        Example:
            ```python
            # Get all emotional relationships
            emotional_rels = graph_manager.get_relationships_by_dimension("emotional")

            # Get high-valence emotional relationships (0.5 to 1.0)
            positive_rels = graph_manager.get_relationships_by_dimension(
                "emotional", valence_range=(0.5, 1.0)
            )
            ```
        """
        results = []

        for source_id, target_id, data in self.g.edges(data=True):
            edge_dimension = data.get("dimension", "lexical")
            edge_rel_type = data.get("relationship", "related")  # Consistent key

            # --- Dimension Filtering ---
            match = False
            if edge_dimension == dimension:
                match = True
            # Special case: Include meta-emotional links when requesting lexical?
            # This logic seems potentially confusing. Let's refine:
            # If include_meta is True, we might want *any* edge involving the node,
            # regardless of the *edge's* dimension, if the *node* fits other criteria.
            # However, the current signature filters by *edge* dimension.
            # Let's stick to filtering by edge dimension for clarity,
            # and suggest separate methods for node-centric meta-analysis.
            # The original 'include_meta' logic was ambiguous. Removing it for now.
            # if include_meta and dimension == "lexical" and edge_dimension == "emotional":
            #     match = True # Include emotional edges if include_meta and asking for lexical?

            if not match:
                continue

            # --- Relationship Type Filtering ---
            if rel_type and edge_rel_type != rel_type:
                continue

            # --- Valence Range Filtering (only if dimension is emotional) ---
            if valence_range and edge_dimension == "emotional":
                # Check source node's valence
                source_valence = self.g.nodes[source_id].get("valence")
                if source_valence is not None:
                    try:
                        val = float(source_valence)
                        min_val, max_val = valence_range
                        if not (min_val <= val <= max_val):
                            continue  # Skip if source valence is outside range
                    except (ValueError, TypeError):
                        logging.warning(
                            f"Invalid valence '{source_valence}' for node {source_id}. Skipping valence filter."
                        )
                else:
                    continue  # Skip if source node has no valence for filtering

            # --- Collect Data ---
            source_term = self.g.nodes[source_id].get("term", f"Unknown_{source_id}")
            target_term = self.g.nodes[target_id].get("term", f"Unknown_{target_id}")

            # Prepare attributes dictionary (copy edge data)
            attributes = data.copy()

            results.append((str(source_term), str(target_term), str(edge_rel_type), attributes))

        return results

    @eidosian()
    def get_emotional_subgraph(
        self,
        term: str,
        depth: int = 1,
        context: Optional[Union[str, Dict[str, float]]] = None,
        emotional_types: Optional[List[str]] = None,
        min_intensity: float = 0.0,  # Default to include all intensities >= 0
    ) -> nx.Graph:
        """
        Extract a subgraph of emotional relationships centered on a specific term.

        Builds a subgraph containing the central term and emotionally related terms
        within a specified depth. Allows filtering by emotional context, specific
        relationship types, and minimum intensity (weight).

        Args:
            term: The central term to build the subgraph around.
            depth: The number of relationship steps (hops) to include (default: 1).
            context: Optional emotional context name (str) previously integrated,
                     or a dictionary of context factors and weights (e.g., {"clinical": 0.8}).
                     If provided, edge weights might be adjusted based on context relevance.
            emotional_types: Optional list of specific emotional relationship types
                             to include (e.g., ["intensifies", "evokes"]). If None,
                             all emotional dimension edges are considered initially.
            min_intensity: Minimum edge weight (intensity) for relationships to be
                           included in the subgraph (default: 0.0).

        Returns:
            A NetworkX graph containing the emotional subgraph. Nodes include
            emotional metadata (valence, arousal) if available, and 'emotional_centrality'.
            Edges may have adjusted 'weight' based on context.

        Raises:
            NodeNotFoundError: If the term is not found in the graph.
            GraphError: If subgraph extraction or processing fails.

        Example:
            ```python
            # Get emotional neighborhood of "happiness" (depth 2)
            emo_graph = graph_manager.get_emotional_subgraph("happiness", depth=2)

            # Get intense clinical context relationships for "anxiety"
            clinical_graph = graph_manager.get_emotional_subgraph(
                "anxiety", context="clinical", min_intensity=0.5
            )
            ```
        """
        term_lower = term.lower()
        if term_lower not in self._term_to_id:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph")

        center_id = self._term_to_id[term_lower]
        if center_id not in self.g:
            raise NodeNotFoundError(f"Center node ID {center_id} for term '{term}' not found in graph nodes.")

        # --- Node Selection (BFS on Emotional Edges) ---
        nodes_to_include: Set[int] = {center_id}
        frontier: Set[int] = {center_id}

        for current_depth in range(depth):
            next_frontier: Set[int] = set()
            if not frontier:
                break  # Stop if no new nodes were added in the previous step

            for node in frontier:
                if node not in self.g:
                    continue  # Skip if node somehow disappeared

                for neighbor in self.g.neighbors(node):
                    if neighbor in nodes_to_include:
                        continue  # Already included

                    edge_data = self.g.get_edge_data(node, neighbor)
                    if not edge_data:
                        continue  # Skip if edge data is missing

                    # Filter 1: Must be an emotional dimension edge
                    if edge_data.get("dimension") != "emotional":
                        continue

                    # Filter 2: Check emotional relationship type if filter is active
                    rel_type = edge_data.get("relationship", "")
                    if emotional_types and rel_type not in emotional_types:
                        continue

                    # Filter 3: Check minimum intensity/weight
                    # Use 'intensity' if present, otherwise 'weight'
                    weight = float(edge_data.get("weight", 0.0))
                    intensity = float(edge_data.get("intensity", weight))  # Fallback to weight
                    if intensity < min_intensity:
                        continue

                    # If all filters pass, add neighbor to the next frontier
                    next_frontier.add(neighbor)

            nodes_to_include.update(next_frontier)
            frontier = next_frontier

        # --- Subgraph Creation and Edge Filtering ---
        try:
            # Create the subgraph based on selected nodes
            emotional_subgraph = self.g.subgraph(nodes_to_include).copy()
        except Exception as e:
            raise GraphError(f"Failed to create subgraph from selected nodes: {e}", e) from e

        # --- Context Application and Final Edge Filtering ---
        context_factors: Dict[str, float] = {}
        if isinstance(context, str):
            # Retrieve pre-defined context weights
            context_factors = self._emotional_contexts.get(context, {})
            if not context_factors:
                logging.warning(f"Emotional context '{context}' not found. Proceeding without context weighting.")
        elif isinstance(context, dict):
            context_factors = context

        edges_to_remove = []
        for u, v, data in emotional_subgraph.edges(data=True):
            # Re-apply filters strictly to edges *within* the subgraph
            if data.get("dimension") != "emotional":
                edges_to_remove.append((u, v))
                continue
            if emotional_types and data.get("relationship", "") not in emotional_types:
                edges_to_remove.append((u, v))
                continue
            weight = float(data.get("weight", 0.0))
            intensity = float(data.get("intensity", weight))
            if intensity < min_intensity:
                edges_to_remove.append((u, v))
                continue

            # Apply context weighting if context factors are defined
            if context_factors:
                data["original_weight"] = weight  # Store original weight
                context_relevance = 0.0
                # Simple relevance: check if context factors appear in relationship type or associated terms
                rel_type = data.get("relationship", "").lower()
                term_u = emotional_subgraph.nodes[u].get("term", "").lower()
                term_v = emotional_subgraph.nodes[v].get("term", "").lower()

                for factor, factor_weight in context_factors.items():
                    factor_lower = factor.lower()
                    if factor_lower in rel_type or factor_lower in term_u or factor_lower in term_v:
                        # Use the highest weight found if multiple factors match
                        context_relevance = max(context_relevance, factor_weight)

                # Adjust weight based on relevance (e.g., simple multiplication)
                # More sophisticated weighting could be applied here.
                # Example: Boost weight by (1 + relevance)
                adjusted_weight = weight * (1.0 + context_relevance)
                data["weight"] = adjusted_weight
                data["context_relevance"] = context_relevance  # Store relevance score

        emotional_subgraph.remove_edges_from(edges_to_remove)

        # --- Node Metadata Enhancement ---
        subgraph_degrees = dict(emotional_subgraph.degree())
        for node_id in emotional_subgraph.nodes():
            # Add emotional metadata (valence, arousal) if available in original graph
            original_node_data = self.g.nodes.get(node_id, {})
            emotional_subgraph.nodes[node_id]["valence"] = original_node_data.get("valence")
            emotional_subgraph.nodes[node_id]["arousal"] = original_node_data.get("arousal")

            # Add degree centrality within the emotional subgraph
            emotional_subgraph.nodes[node_id]["emotional_centrality"] = subgraph_degrees.get(node_id, 0)

        return emotional_subgraph

    @eidosian()
    def analyze_multidimensional_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships across different dimensions with emotional intelligence.

        Provides comprehensive statistics about relationship types across lexical,
        emotional, and affective dimensions, identifying patterns, correlations,
        emotional clusters, meta-emotional relationships, and emotional transitions.

        Returns:
            Dict[str, Any]: Dictionary with detailed analysis including:
            - dimensions (Dict[str, int]): Counts of relationships by dimension.
            - co_occurrences (Dict[str, int]): Co-occurrence counts of dimension pairs on nodes.
            - most_common (Dict[str, List[Tuple[str, int]]]): Top 5 relationship types per dimension.
            - multi_dimensional_nodes (Dict[str, Dict[str, Any]]): Terms connected via multiple dimensions.
            - emotional_valence_distribution (Dict[str, Any]): Stats on emotional valence (mean, median, etc.).
            - meta_emotional_patterns (Dict[str, List[str]]): Source emotions linking to target emotions via meta-rels.
            - emotional_clusters (List[Dict[str, Any]]): List of detected emotional clusters (if library available).
            - affective_transitions (List[Tuple[str, str, float]]): Top 10 potential transitions based on valence change and weight.

        Example:
            ```python
            # Get comprehensive multidimensional analysis
            analysis = graph_manager.analyze_multidimensional_relationships()

            # Examine emotional dimensions
            emotional_stats = analysis.get("dimensions", {}).get("emotional", 0)
            print(f"Emotional relationships: {emotional_stats}")

            # Examine meta-emotional patterns
            meta_patterns = analysis.get("meta_emotional_patterns", {})
            for source, targets in meta_patterns.items():
                print(f"Meta-emotion: {source} → {', '.join(targets)}")

            # Examine emotional transitions
            transitions = analysis.get("affective_transitions", [])
            for source, target, strength in transitions[:5]:  # Show top 5
                print(f"Emotional transition: {source} → {target} (strength: {strength:.2f})")
            ```
        """
        results: Dict[str, Any] = {
            "dimensions": {},
            "co_occurrences": {},
            "most_common": {},
            "multi_dimensional_nodes": {},
            "emotional_valence_distribution": {},
            "meta_emotional_patterns": {},
            "emotional_clusters": [],
            "affective_transitions": [],
        }

        # 1. Count relationships by dimension
        dimension_counts: Dict[str, int] = {}
        for _, _, data in self.g.edges(data=True):
            dimension = data.get("dimension", "lexical")
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
        results["dimensions"] = dimension_counts

        # 2. Find multi-dimensional nodes, track valence, find meta-emotional sources
        nodes_with_multi_dimensions: Dict[str, Dict[str, Any]] = {}
        valence_values: List[float] = []
        meta_emotional_sources: Dict[str, List[str]] = {}  # Map source term -> list of target terms

        meta_relationship_types = {
            "meta_emotion",
            "evokes",
            "emotional_component",
            "intensifies",
            "diminishes",
        }

        for node_id, node_data in self.g.nodes(data=True):
            term = node_data.get("term")
            if not term or not isinstance(term, str):
                continue  # Skip nodes without valid terms

            node_dimensions: Set[str] = set()
            emotional_neighbors_info: List[Tuple[str, str]] = []  # (neighbor_term, rel_type)

            # Track valence
            valence = node_data.get("valence")
            if valence is not None:
                try:
                    valence_values.append(float(valence))
                except (ValueError, TypeError):
                    pass  # Ignore invalid valence values

            # Examine neighbors
            for neighbor_id in self.g.neighbors(node_id):
                if neighbor_id not in self.g:
                    continue  # Ensure neighbor exists
                edge_data = self.g.get_edge_data(node_id, neighbor_id)
                if not edge_data:
                    continue

                dimension = edge_data.get("dimension", "lexical")
                node_dimensions.add(dimension)

                rel_type = edge_data.get("relationship", "")
                neighbor_term = self.g.nodes[neighbor_id].get("term")
                if not neighbor_term or not isinstance(neighbor_term, str):
                    continue  # Skip neighbors without valid terms

                # Track emotional connections
                if dimension == "emotional":
                    emotional_neighbors_info.append((neighbor_term, rel_type))

                # Check for meta-emotional relationships
                if rel_type in meta_relationship_types:
                    # Ensure both source and target are considered 'emotional' (e.g., have valence)
                    neighbor_valence = self.g.nodes[neighbor_id].get("valence")
                    if valence is not None and neighbor_valence is not None:  # Both nodes should be emotional
                        if term not in meta_emotional_sources:
                            meta_emotional_sources[term] = []
                        if neighbor_term not in meta_emotional_sources[term]:  # Avoid duplicates
                            meta_emotional_sources[term].append(neighbor_term)

            # Record nodes with multiple dimensions
            if len(node_dimensions) > 1:
                nodes_with_multi_dimensions[term] = {
                    "dimensions": sorted(list(node_dimensions)),
                    "emotional_connections": sorted(emotional_neighbors_info),
                }

        results["multi_dimensional_nodes"] = nodes_with_multi_dimensions
        results["meta_emotional_patterns"] = {
            k: sorted(v) for k, v in meta_emotional_sources.items()
        }  # Sort for consistency

        # 3. Calculate emotional valence distribution
        if valence_values:
            results["emotional_valence_distribution"] = {
                "count": len(valence_values),
                "mean": float(np.mean(valence_values)),
                "median": float(np.median(valence_values)),
                "std": float(np.std(valence_values)),
                "min": float(min(valence_values)),
                "max": float(max(valence_values)),
                "positive_ratio": sum(1 for v in valence_values if v > 0) / len(valence_values),
                "negative_ratio": sum(1 for v in valence_values if v < 0) / len(valence_values),
                "neutral_ratio": sum(1 for v in valence_values if v == 0) / len(valence_values),
            }
        else:
            results["emotional_valence_distribution"] = {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
            }

        # 4. Find most common relationship types by dimension
        relationship_by_dimension: Dict[str, Dict[str, int]] = {}
        for _, _, data in self.g.edges(data=True):
            dimension = data.get("dimension", "lexical")
            rel_type = data.get("relationship", "related")
            if not isinstance(rel_type, str):
                continue  # Ensure rel_type is string

            if dimension not in relationship_by_dimension:
                relationship_by_dimension[dimension] = {}

            relationship_by_dimension[dimension][rel_type] = relationship_by_dimension[dimension].get(rel_type, 0) + 1

        for dimension, counts in relationship_by_dimension.items():
            sorted_rels = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            results["most_common"][dimension] = sorted_rels[:5]  # Top 5

        # 5. Analyze co-occurrences between dimensions on nodes
        dimension_pairs: Dict[str, int] = {}
        for (
            term,
            node_info,
        ) in nodes_with_multi_dimensions.items():  # Use pre-calculated multi-dim nodes
            dimensions_present = node_info.get("dimensions", [])
            # Generate unique pairs
            for i in range(len(dimensions_present)):
                for j in range(i + 1, len(dimensions_present)):
                    # Sort pair alphabetically for consistent key
                    dim_pair = tuple(sorted((dimensions_present[i], dimensions_present[j])))
                    key = f"{dim_pair[0]}_&_{dim_pair[1]}"
                    dimension_pairs[key] = dimension_pairs.get(key, 0) + 1
        results["co_occurrences"] = dimension_pairs

        # 6. Identify emotional clusters (using existing semantic cluster analysis, filtered)
        try:
            # Reuse analyze_semantic_clusters with emotion_only=True
            emotional_clusters_raw = self.analyze_semantic_clusters(emotion_only=True)
            # Reformat slightly if needed, or use as is
            results["emotional_clusters"] = [
                {"id": cid, "terms": terms, "size": len(terms)} for cid, terms in emotional_clusters_raw.items()
            ]
        except ImportError:
            logging.warning("Louvain community library not found. Skipping emotional cluster analysis.")
            results["emotional_clusters"] = []  # Ensure key exists
        except Exception as cluster_err:
            logging.error(f"Error during emotional cluster analysis: {cluster_err}")
            results["emotional_clusters"] = []  # Ensure key exists

        # 7. Identify common affective transitions based on valence change
        transitions: List[Tuple[str, str, float]] = []
        emotional_node_ids = {node_id for node_id, data in self.g.nodes(data=True) if data.get("valence") is not None}

        for source_id in emotional_node_ids:
            source_term = self.g.nodes[source_id].get("term", "")
            source_valence = self.g.nodes[source_id].get("valence", 0.0)  # Default to 0.0 if missing after check

            for target_id in self.g.neighbors(source_id):
                if target_id in emotional_node_ids:  # Target must also be emotional
                    edge_data = self.g.get_edge_data(source_id, target_id)
                    if edge_data and edge_data.get("dimension") == "emotional":
                        target_term = self.g.nodes[target_id].get("term", "")
                        target_valence = self.g.nodes[target_id].get("valence", 0.0)

                        # Consider transitions only if valence changes significantly (optional threshold)
                        valence_diff = target_valence - source_valence
                        if abs(valence_diff) > 0.01:  # Avoid floating point noise for zero change
                            # Calculate transition strength (e.g., edge weight * magnitude of change)
                            weight = float(edge_data.get("weight", 1.0))
                            transition_strength = weight * abs(valence_diff)
                            transitions.append(
                                (
                                    str(source_term),
                                    str(target_term),
                                    float(transition_strength),
                                )
                            )

        # Sort transitions by strength and take top 10
        if transitions:
            results["affective_transitions"] = sorted(transitions, key=lambda x: x[2], reverse=True)[:10]

        return results

    @eidosian()
    def extract_meta_emotional_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract patterns of meta-emotions (emotions about emotions) from the graph.

        Identifies and analyzes meta-emotional relationships, where one emotional term
        relates to another emotional term through relationships like "evokes", "meta_emotion",
        or "emotional_component". These patterns reveal how emotions relate to each other
        in complex hierarchies and networks.

        Returns:
            Dictionary mapping source emotions to lists of target emotion dictionaries,
            each containing the target term, relationship type, and relationship strength

        Example:
            ```python
            # Extract meta-emotional patterns
            patterns = graph_manager.extract_meta_emotional_patterns()

            # Examine patterns
            for source_emotion, targets in patterns.items():
                print(f"Emotion: {source_emotion}")
                for target in targets:
                    print(f"  → {target['term']} ({target['relationship']})"
                          f" strength: {target['strength']:.2f}")
            ```
        """
        patterns = {}

        # Find all emotional terms (nodes with "emotion" in their description or tagged as emotional)
        emotional_nodes = []
        for node, attrs in self.g.nodes(data=True):
            term = attrs.get("term", "")
            if not term:
                continue

            # Check if node has valence/arousal or is marked as emotional
            has_emotional_attrs = attrs.get("valence") is not None or attrs.get("arousal") is not None

            # Also check node's edges for emotional connections
            has_emotional_edges = any(
                self.g.get_edge_data(node, neighbor).get("dimension") == "emotional"
                for neighbor in self.g.neighbors(node)
            )

            if has_emotional_attrs or has_emotional_edges:
                emotional_nodes.append(node)

        # Analyze connections between emotional nodes
        meta_relationships = [
            "meta_emotion",
            "evokes",
            "emotional_component",
            "intensifies",
            "diminishes",
        ]

        for source in emotional_nodes:
            source_term = self.g.nodes[source].get("term", "")

            # Find meta-emotional connections
            meta_targets = []

            for target in self.g.neighbors(source):
                if target in emotional_nodes:
                    edge_data = self.g.get_edge_data(source, target)
                    rel_type = edge_data.get("relationship", "")

                    # Check if this is a meta-emotional relationship
                    if rel_type in meta_relationships:
                        target_term = self.g.nodes[target].get("term", "")
                        weight = edge_data.get("weight", 1.0)

                        meta_targets.append(
                            {
                                "term": target_term,
                                "relationship": rel_type,
                                "strength": weight,
                                "bidirectional": edge_data.get("bidirectional", False),
                                "source_valence": self.g.nodes[source].get("valence"),
                                "target_valence": self.g.nodes[target].get("valence"),
                            }
                        )

            # Only include emotions with meta-connections
            if meta_targets:
                patterns[source_term] = meta_targets

        return patterns

    @eidosian()
    def analyze_emotional_valence_distribution(self, dimension: str = "emotional"):
        """
        Analyze the distribution of emotional valence across the graph.

        Computes statistical metrics about the emotional valence (positivity/negativity)
        of terms in the graph, provides distribution histograms, and identifies terms at
        the extremes of the valence spectrum.

        Args:
            dimension: Dimension to analyze, typically "emotional" (default) or "affective"

        Returns:
            Dictionary with statistical analysis including:
            - count: Number of terms with valence values
            - mean/median/std: Statistical measures of central tendency and spread
            - range: Minimum and maximum valence values
            - distribution: Histogram of valence value frequencies
            - top_positive/top_negative: Terms with highest/lowest valence scores
            - clusters: Valence-based term clusters

        Example:
            ```python
            # Analyze emotional valence distribution
            valence_stats = graph_manager.analyze_emotional_valence_distribution()

            # Print key statistics
            print(f"Emotional terms: {valence_stats['count']}")
            print(f"Mean valence: {valence_stats['mean']:.2f}")
            print(f"Valence range: {valence_stats['range'][0]:.2f} to {valence_stats['range'][1]:.2f}")

            # Print most positive and negative terms
            print("Most positive terms:")
            for term, val in valence_stats['top_positive']:
                print(f"  - {term}: {val:.2f}")

            print("Most negative terms:")
            for term, val in valence_stats['top_negative']:
                print(f"  - {term}: {val:.2f}")
            ```
        """
        # Collection stage
        valence_data = []
        term_to_valence = {}

        for node, attrs in self.g.nodes(data=True):
            term = attrs.get("term", "")
            if not term:
                continue

            # Try to get valence from node attributes
            valence = attrs.get("valence")

            # If not found directly, check if this node has emotional edges
            if valence is None:
                # Find average valence from emotional edges
                emotional_edges = [
                    self.g.get_edge_data(node, neighbor)
                    for neighbor in self.g.neighbors(node)
                    if self.g.get_edge_data(node, neighbor).get("dimension") == dimension
                ]

                if emotional_edges:
                    # Extract valence values, default to 0 if not specified
                    edge_valences = [edge.get("valence", 0) for edge in emotional_edges if "valence" in edge]

                    if edge_valences:
                        valence = sum(edge_valences) / len(edge_valences)

            # If we found a valence value, record it
            if valence is not None:
                valence_data.append(valence)
                term_to_valence[term] = valence

        # Statistical analysis
        if not valence_data:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "range": [0, 0],
                "distribution": {},
                "top_positive": [],
                "top_negative": [],
                "clusters": {},
            }

        # Calculate basic statistics
        count = len(valence_data)
        mean_valence = np.mean(valence_data)
        median_valence = np.median(valence_data)
        std_valence = np.std(valence_data)
        min_valence = min(valence_data)
        max_valence = max(valence_data)

        # Generate histogram (distribution)
        hist_bins = 7  # -1.0 to 1.0 in increments of ~0.3
        hist, bin_edges = np.histogram(valence_data, bins=hist_bins, range=(-1.0, 1.0))

        # Convert histogram to dictionary
        distribution = {}
        for i, count in enumerate(hist):
            bin_name = f"{bin_edges[i]:.1f} to {bin_edges[i+1]:.1f}"
            distribution[bin_name] = int(count)

        # Find top positive and negative terms
        sorted_terms = sorted(term_to_valence.items(), key=lambda x: x[1])
        top_negative = sorted_terms[: min(5, len(sorted_terms) // 5)]  # Bottom 20% up to 5 terms
        top_positive = sorted_terms[-min(5, len(sorted_terms) // 5) :]  # Top 20% up to 5 terms

        # Group terms into clusters by valence range
        clusters = {
            "very_negative": [],  # -1.0 to -0.6
            "negative": [],  # -0.6 to -0.2
            "slightly_negative": [],  # -0.2 to -0.0
            "neutral": [],  # 0.0
            "slightly_positive": [],  # 0.0 to 0.2
            "positive": [],  # 0.2 to 0.6
            "very_positive": [],  # 0.6 to 1.0
        }

        for term, val in term_to_valence.items():
            if val < -0.6:
                clusters["very_negative"].append(term)
            elif val < -0.2:
                clusters["negative"].append(term)
            elif val < 0:
                clusters["slightly_negative"].append(term)
            elif val == 0:
                clusters["neutral"].append(term)
            elif val <= 0.2:
                clusters["slightly_positive"].append(term)
            elif val <= 0.6:
                clusters["positive"].append(term)
            else:
                clusters["very_positive"].append(term)

        # Final result structure
        return {
            "count": count,
            "mean": mean_valence,
            "median": median_valence,
            "std": std_valence,
            "range": [min_valence, max_valence],
            "distribution": distribution,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "clusters": {k: v for k, v in clusters.items() if v},  # Only include non-empty clusters
        }

    @eidosian()
    def integrate_emotional_context(self, context_name: str, context_weights: Dict[str, float]) -> int:
        """
        Integrate an emotional context into the graph for contextual analysis.

        Registers an emotional context definition that can be used to filter
        and weight relationships during emotional analysis. This allows for
        domain-specific or situation-specific emotional interpretations.

        Args:
            context_name: Name to assign to this emotional context
            context_weights: Dictionary of emotional factors and their weights
                           in this context (e.g., {'clinical': 0.8, 'urgency': 0.6})

        Returns:
            int: Count of updated edges

        Example:
            ```python
            # Define a clinical medical context
            graph_manager.integrate_emotional_context(
                "clinical",
                {
                    "professional": 0.9,
                    "detached": 0.7,
                    "analytical": 0.8,
                    "urgency": 0.6,
                    "empathy": 0.4
                }
            )

            # Define a literary context
            graph_manager.integrate_emotional_context(
                "literary",
                {
                    "expressive": 0.9,
                    "narrative": 0.8,
                    "descriptive": 0.7,
                    "dramatic": 0.6
                }
            )

            # Later use the context in analysis
            subgraph = graph_manager.get_emotional_subgraph(
                "anxiety", context="clinical"
            )
            ```
        """
        # Store the context in graph metadata
        if not hasattr(self, "_emotional_contexts"):
            self._emotional_contexts = {}

        # Validate weights (should be between 0 and 1)
        for factor, weight in context_weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Context weight for '{factor}' must be between 0 and 1")

        # Store the context
        self._emotional_contexts[context_name] = context_weights

        # Add context as a graph attribute for persistence
        self.g.graph[f"emotional_context_{context_name}"] = context_weights

        # Apply context tags to relevant emotional edges
        # This is a simplified implementation - in a real system, you would use
        # more sophisticated matching to determine which edges relate to this context
        updated_count = 0

        for source, target, data in self.g.edges(data=True):
            if data.get("dimension") == "emotional":
                # Check if any context factors match this relationship
                for factor in context_weights.keys():
                    # Very simple matching - check if factor appears in relationship type
                    rel_type = data.get("relationship", "")
                    source_term = self.g.nodes[source].get("term", "").lower()
                    target_term = self.g.nodes[target].get("term", "").lower()

                    # Simple heuristic - if factor is in the relationship or terms
                    if (
                        factor.lower() in rel_type.lower()
                        or factor.lower() in source_term
                        or factor.lower() in target_term
                    ):

                        # Tag edge with context
                        data["context"] = context_name
                        # Adjust weight based on context importance
                        data["context_weight"] = context_weights[factor]
                        updated_count += 1
                        break

        return updated_count

    @eidosian()
    def analyze_emotional_transitions(
        self, path_length: int = 2, min_transition_strength: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Analyze emotional transition pathways in the emotion graph.

        Identifies common paths of emotional change/transition, showing how
        emotions flow from one state to another. These transitions reveal
        emotional narratives, progression patterns, and psychological pathways.

        Args:
            path_length: Maximum path length to consider (default: 2)
            min_transition_strength: Minimum strength threshold for transitions

        Returns:
            List of dictionaries representing transition paths, each containing:
            - path: List of terms in the transition path
            - strength: Overall transition strength
            - valence_shift: Net change in valence
            - arousal_shift: Net change in arousal
            - relationship_types: Types of relationships in the path

        Example:
            ```python
            # Analyze emotional transitions
            transitions = graph_manager.analyze_emotional_transitions()

            # Print top transitions
            for t in transitions[:5]:  # Top 5
                path_str = " → ".join(t["path"])
                print(f"Transition: {path_str}")
                print(f"  Strength: {t['strength']:.2f}")
                print(f"  Valence shift: {t['valence_shift']:.2f}")
                print(f"  Relationship types: {', '.join(t['relationship_types'])}")
            ```
        """
        # Find emotional nodes
        emotional_nodes = []
        for node, attrs in self.g.nodes(data=True):
            valence = attrs.get("valence")
            arousal = attrs.get("arousal")

            # Include node if it has emotional attributes
            if valence is not None or arousal is not None:
                emotional_nodes.append(node)

        # If too few emotional nodes, return empty result
        if len(emotional_nodes) < 2:
            return []

        transitions = []

        # Analyze paths between emotional nodes
        for source in emotional_nodes:
            source_valence = self.g.nodes[source].get("valence", 0)
            source_arousal = self.g.nodes[source].get("arousal", 0)

            # Use simple BFS to find paths up to path_length
            paths = self._find_emotional_paths(source, path_length, emotional_nodes)

            for path in paths:
                # Skip too short paths
                if len(path) < 2:
                    continue

                # Calculate transition properties
                path_terms = [self.g.nodes[n].get("term", "") for n in path]
                relationship_types = []

                # Calculate cumulative strength and changes
                strength = 1.0  # Start with full strength

                # Get initial and final emotional states
                final_node = path[-1]
                final_valence = self.g.nodes[final_node].get("valence", 0)
                final_arousal = self.g.nodes[final_node].get("arousal", 0)

                # Calculate shifts
                valence_shift = final_valence - source_valence
                arousal_shift = final_arousal - source_arousal

                # Extract relationship types along the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.g.get_edge_data(u, v)
                    rel_type = edge_data.get("relationship", "related")
                    relationship_types.append(rel_type)

                    # Multiply by edge weight to get path strength
                    edge_weight = edge_data.get("weight", 1.0)
                    strength *= edge_weight

                # Only include paths above threshold
                if strength >= min_transition_strength:
                    transitions.append(
                        {
                            "path": path_terms,
                            "strength": strength,
                            "valence_shift": valence_shift,
                            "arousal_shift": arousal_shift,
                            "relationship_types": relationship_types,
                        }
                    )

        # Sort transitions by strength
        transitions.sort(key=lambda x: x["strength"], reverse=True)

        return transitions

    def _find_emotional_paths(
        self, start_node: WordId, max_length: int, valid_nodes: List[WordId]
    ) -> List[List[WordId]]:
        """
        Find paths from start_node to other emotional nodes up to max_length.

        Helper method for analyze_emotional_transitions().

        Args:
            start_node: Starting node ID
            max_length: Maximum path length
            valid_nodes: List of valid destination nodes (emotional nodes)

        Returns:
            List of node ID paths
        """
        # Simple BFS path finding
        paths: List[List[WordId]] = []
        queue: List[Tuple[WordId, List[WordId]]] = [(start_node, [start_node])]

        while queue:
            node, path = queue.pop(0)

            # If path is already at max length, don't expand further
            if len(path) >= max_length:
                continue

            for neighbor in self.g.neighbors(node):
                # Cast neighbor to WordId to satisfy type checker
                neighbor_id: WordId = cast(WordId, neighbor)

                # Skip already visited nodes in this path
                if neighbor_id in path:
                    continue

                # Create new path with this neighbor
                new_path: List[WordId] = path + [neighbor_id]

                # If neighbor is a valid destination, add the path
                if neighbor_id in valid_nodes and neighbor_id != start_node:
                    paths.append(new_path)

                # Continue BFS with this new path
                queue.append((neighbor_id, new_path))

        return paths

    @eidosian()
    def verify_database_tables(self) -> bool:
        """Verify that required database tables exist.

        Returns:
            bool: True if tables exist, False otherwise
        """
        try:
            # Use _db_connection to verify tables
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Check if words table exists
                cursor.execute(database_manager.SQL_CHECK_WORDS_TABLE)
                if not cursor.fetchone():
                    return False

                # Check if relationships table exists
                cursor.execute(database_manager.SQL_CHECK_RELATIONSHIPS_TABLE)
                if not cursor.fetchone():
                    return False

                return True
        except Exception as e:
            # Log the specific database error for debugging
            logging.error(f"Database verification failed: {e}")
            return False
