"""
Central orchestrator for knowledge graph operations in Word Forge.

Manages the lifecycle, structure, analysis, and visualization of a
multidimensional knowledge graph representing lexical, emotional, and
affective relationships between terms. Integrates functionalities from
specialized sub-modules (Builder, Query, Layout, Visualizer, IO, Analysis).

Adheres to Eidosian principles: modularity, precision, recursive refinement,
and structural integrity. Ensures thread-safe operations where applicable.

Architecture:
    ┌────────────────────┐
    │    GraphManager    │ (Orchestrator)
    │ (State & Interface)│
    └─────────┬──────────┘
              │ Delegates To:
    ┌─────────┴───────────────────────────────────────────────┐
    │         │           │          │            │           │
┌───▼────┐ ┌──▼─────┐ ┌───▼────┐ ┌───▼──────┐ ┌───▼───┐ ┌───▼──────┐
│ Builder│ │ Query  │ │ Layout │ │Visualizer│ │  IO   │ │ Analysis │
│(DB→Graph)│(Info/Srch)│(Position)│ (Plotting) │(Files)│ (Insights) │
└────────┘ └────────┘ └────────┘ └──────────┘ └───────┘ └──────────┘
      │         │          │          │           │         │
      └─────────┴──────────┴──────────┴───────────┴─────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │    NetworkX    │ (Core Graph Lib)
                      └────────────────┘
                      ┌────────────────┐
                      │   DBManager    │ (Persistence)
                      └────────────────┘
                      ┌────────────────┐
                      │ Optional Libs  │ (Pyvis, Plotly, etc.)
                      └────────────────┘
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import networkx as nx
from eidosian_core import eidosian

# Core components
from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.exceptions import NodeNotFoundError
from word_forge.graph.graph_analysis import (
    ClusterResult,
    GraphAnalysis,
    MetaEmotionalResult,
    MultiDimResult,
    TransitionResult,
    ValenceDistResult,
)
from word_forge.graph.graph_builder import GraphBuilder, GraphUpdateMetrics
from word_forge.graph.graph_config import (
    GraphConfig,
    GraphInfoDict,
    LayoutAlgorithm,
    PositionDict,  # Ensure PositionDict is imported
    RelationshipDimension,
    RelType,
    Term,
    WordId,
)
from word_forge.graph.graph_io import GraphIO
from word_forge.graph.graph_layout import GraphLayout
from word_forge.graph.graph_query import GraphQuery
from word_forge.graph.graph_visualizer import GraphVisualizer

# Make relationship_properties accessible if needed internally
from word_forge.relationships import RELATIONSHIP_TYPES as relationship_properties
from word_forge.relationships import RelationshipProperties


class GraphManager:
    """
    Orchestrates knowledge graph operations, managing state and sub-modules.

    Provides a unified interface for building, querying, analyzing, visualizing,
    and managing the lifecycle of the multidimensional knowledge graph.

    Attributes:
        db_manager: Instance of DBManager for database interactions.
        config: Graph configuration settings.
        g: The core NetworkX graph object (Graph or DiGraph).
        dimensions: Dimensionality for layout and visualization (2 or 3).
        builder: Instance of GraphBuilder.
        query: Instance of GraphQuery.
        layout: Instance of GraphLayout.
        visualizer: Instance of GraphVisualizer.
        io: Instance of GraphIO.
        analysis: Instance of GraphAnalysis.
        logger: Logger instance for this manager.
    """

    def __init__(
        self,
        db_manager: DBManager,
        graph_config: Optional[GraphConfig] = None,
        graph_type: type = nx.Graph,  # Default to undirected graph
        dimensions: int = 2,
    ) -> None:
        """
        Initialize the GraphManager.

        Args:
            db_manager: The database manager instance.
            graph_config: Optional graph configuration. Defaults to global config.
            graph_type: The type of NetworkX graph to use (nx.Graph or nx.DiGraph).
            dimensions: The dimensionality for layout/visualization (2 or 3).
        """
        self.db_manager = db_manager
        self.config = graph_config or config.graph
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GraphManager...")

        # --- Core Graph State ---
        self.g: Union[nx.Graph, nx.DiGraph] = graph_type()
        self.dimensions: int = dimensions
        self._term_to_id: Dict[str, WordId] = {}  # term (lowercase) -> node_id mapping
        self._positions: PositionDict = {}  # node_id -> position tuple mapping
        self._relationship_counts: Counter[RelType] = Counter()
        self._emotional_contexts: Dict[str, Dict[str, float]] = {}  # Stored contexts

        # --- Lock for thread safety on graph modifications ---
        # RLock allows re-entrant locking within the same thread
        self._graph_lock = threading.RLock()

        # --- Initialize Sub-modules ---
        # Pass self (the manager instance) to each sub-module
        self.builder = GraphBuilder(self)
        self.query = GraphQuery(self)
        self.layout = GraphLayout(self)
        self.visualizer = GraphVisualizer(self)
        self.io = GraphIO(self)
        self.analysis = GraphAnalysis(self)

        self.logger.info("GraphManager initialized successfully.")

    # ==========================================
    # Context Manager for DB Connection
    # ==========================================
    @contextmanager
    def _db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Provide a managed database connection by using the DBManager's context.
        Yields the actual connection object.
        """
        # Directly use the context manager from db_manager to get the connection
        with self.db_manager.get_connection() as conn:
            try:
                yield conn  # Yield the actual connection obtained from the inner context
            except sqlite3.Error as db_err:
                # Optional: Add specific error handling/logging here if needed
                self.logger.error(f"Database operation failed within context: {db_err}", exc_info=True)
                raise  # Re-raise the original error
            # The 'finally' block for cleanup is handled by the 'with' statement
            # managing db_manager.get_connection()

    # ==========================================
    # Public Accessors for Internal State
    # ==========================================
    @eidosian()
    def get_positions(self) -> PositionDict:
        """
        Return a copy of the computed node positions. Thread-safe.

        Returns:
            PositionDict: A dictionary mapping node IDs to their positions.
        """
        with self._graph_lock:
            # Return a copy to prevent external modification
            return self._positions.copy()

    # ==========================================
    # Graph Building & Modification Methods (via Builder)
    # ==========================================
    @eidosian()
    def build_graph(self) -> None:
        """
        Build the graph from the database, replacing the existing graph.

        Delegates to GraphBuilder.build_graph. Thread-safe.

        Raises:
            GraphDataError: If fetching data fails.
            GraphError: For other construction issues.
        """
        with self._graph_lock:
            self.builder.build_graph()

    @eidosian()
    def update_graph(self) -> int:
        """
        Incrementally update the graph with new data from the database.

        Delegates to GraphBuilder.update_graph. Thread-safe.

        Returns:
            int: Number of new nodes added.

        Raises:
            GraphDataError: If fetching data fails.
            GraphError: For other update issues.
        """
        with self._graph_lock:
            return self.builder.update_graph()

    @eidosian()
    def get_last_update_metrics(self) -> GraphUpdateMetrics:
        """Expose metrics from the most recent graph build or update cycle."""

        with self._graph_lock:
            return self.builder.last_update_metrics

    @eidosian()
    def ensure_sample_data(self) -> bool:
        """
        Ensure sample data exists in the database if it's empty.

        Delegates to GraphBuilder.ensure_sample_data.

        Returns:
            bool: True if sample data was added, False otherwise.

        Raises:
            GraphError: If adding sample data fails.
        """
        # This modifies the DB, potentially read by builder, lock if needed
        # Although builder methods lock graph access, DB access might need coordination
        # For simplicity here, assuming DB operations are safe or handled by DBManager
        return self.builder.ensure_sample_data()

    @eidosian()
    def verify_database_tables(self) -> bool:
        """
        Verify that required database tables exist.

        Delegates to GraphBuilder.verify_database_tables.

        Returns:
            bool: True if required tables exist, False otherwise.
        """
        return self.builder.verify_database_tables()

    @eidosian()
    def add_word_node(self, term: Term, attributes: Optional[Dict[str, Any]] = None) -> WordId:
        """
        Add a single word node to the graph if it doesn't exist.

        Handles case-insensitive term checking and updates internal mappings.
        Triggers incremental layout update. Thread-safe.

        Args:
            term: The word or phrase to add.
            attributes: Optional dictionary of node attributes (e.g., {'valence': 0.5}).

        Returns:
            WordId: The ID of the added or existing node.

        Raises:
            ValueError: If the term is empty or invalid.
        """
        if not term or not isinstance(term, str):
            raise ValueError("Term must be a non-empty string.")

        term_lower = term.lower()
        with self._graph_lock:
            existing_id = self._term_to_id.get(term_lower)
            if existing_id is not None:
                # Node exists, potentially update attributes
                if attributes:
                    nx.set_node_attributes(self.g, {existing_id: attributes})
                    self.logger.debug(f"Updated attributes for existing node '{term}' (ID: {existing_id}).")
                return existing_id
            else:
                # Node doesn't exist, create new ID and add
                # Simple ID strategy: max_id + 1 (ensure graph isn't empty)
                if self.g:
                    nodes = list(self.g.nodes())
                    new_id = max(nodes) + 1 if nodes else 1
                else:
                    new_id = 1

                node_attrs = {"term": term, "id": new_id}
                if attributes:
                    node_attrs.update(attributes)

                self.g.add_node(new_id, **node_attrs)
                self._term_to_id[term_lower] = new_id
                self.logger.info(f"Added new node '{term}' with ID {new_id}.")

                # Trigger incremental layout update for the new node
                self.layout.update_layout_incrementally([new_id])

                return new_id

    @eidosian()
    def add_relationship(
        self,
        source_term_or_id: Union[Term, WordId],
        target_term_or_id: Union[Term, WordId],
        relationship: RelType,
        dimension: Optional[RelationshipDimension] = None,
        weight: Optional[float] = None,
        color: Optional[str] = None,
        bidirectional: Optional[bool] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Add a relationship (edge) between two terms/nodes.

        Handles resolving terms to IDs, determining relationship properties
        (dimension, weight, color, etc.), and adding the edge with attributes.
        Thread-safe.

        Args:
            source_term_or_id: The source term (str) or node ID (int).
            target_term_or_id: The target term (str) or node ID (int).
            relationship: The type of relationship (e.g., 'synonym').
            dimension: Optional dimension override ('lexical', 'emotional', etc.).
                       If None, determined automatically from relationship type.
            weight: Optional weight override for the edge.
            color: Optional color override for the edge.
            bidirectional: Optional override for edge directionality.
            **kwargs: Additional attributes to add to the edge.

        Returns:
            bool: True if the relationship was added successfully, False otherwise
                  (e.g., if nodes don't exist and cannot be added).

        Raises:
            NodeNotFoundError: If a term is provided but cannot be found or added.
            ValueError: If relationship type is invalid.
        """
        if not relationship or not isinstance(relationship, str):
            raise ValueError("Relationship type must be a non-empty string.")

        # Placeholder for future incremental layout support
        # new_nodes_added: List[WordId] = []

        with self._graph_lock:
            # --- Resolve Source Node ---
            if isinstance(source_term_or_id, str):
                source_id = self.query.get_node_id(source_term_or_id)
                if source_id is None:
                    # Option: Add node implicitly or raise error
                    # self.logger.warning(f"Source term '{source_term_or_id}' not found, adding implicitly.")
                    # source_id = self.add_word_node(source_term_or_id)  # add_word_node handles locking
                    raise NodeNotFoundError(f"Source term '{source_term_or_id}' not found.")
            elif isinstance(source_term_or_id, int):
                source_id = source_term_or_id
                if source_id not in self.g.nodes():
                    raise NodeNotFoundError(f"Source node ID {source_id} not found.")
            else:
                raise TypeError("source_term_or_id must be str or int.")

            # --- Resolve Target Node ---
            if isinstance(target_term_or_id, str):
                target_id = self.query.get_node_id(target_term_or_id)
                if target_id is None:
                    # Option: Add node implicitly or raise error
                    # self.logger.warning(f"Target term '{target_term_or_id}' not found, adding implicitly.")
                    # target_id = self.add_word_node(target_term_or_id)  # add_word_node handles locking
                    raise NodeNotFoundError(f"Target term '{target_term_or_id}' not found.")
            elif isinstance(target_term_or_id, int):
                target_id = target_term_or_id
                if target_id not in self.g.nodes():
                    raise NodeNotFoundError(f"Target node ID {target_id} not found.")
            else:
                raise TypeError("target_term_or_id must be str or int.")

            # Prevent self-loops
            if source_id == target_id:
                self.logger.warning(f"Attempted to add self-loop for node {source_id}. Skipped.")
                return False

            # --- Determine Edge Properties ---
            rel_props = self._get_relationship_properties(relationship)
            edge_dimension = dimension or self._determine_dimension(relationship)
            edge_weight = weight if weight is not None else rel_props.get("weight", 1.0)
            edge_color = color or rel_props.get("color", self.config.get_relationship_color(relationship))
            edge_bidirectional = bidirectional if bidirectional is not None else rel_props.get("bidirectional", False)

            # --- Construct Edge Attributes ---
            source_term_text = self.query.get_term_by_id(source_id) or f"ID:{source_id}"
            target_term_text = self.query.get_term_by_id(target_id) or f"ID:{target_id}"

            edge_attrs = {
                "relationship": relationship,
                "weight": edge_weight,
                "color": edge_color,
                "bidirectional": edge_bidirectional,
                "dimension": edge_dimension,
                "title": f"{relationship}: {source_term_text} {'↔' if edge_bidirectional else '→'} {target_term_text}",
                **kwargs,  # Include any additional user-provided attributes
            }

            # --- Add Edge ---
            has_edge = (
                self.g.has_edge(source_id, target_id)
                if hasattr(self.g, "has_edge")
                else (source_id, target_id) in self.g.edges()
            )
            if has_edge:
                nx.set_edge_attributes(self.g, {(source_id, target_id): edge_attrs})
                self.logger.debug(f"Updated existing edge between {source_id} and {target_id}.")
            else:
                self.g.add_edge(source_id, target_id, **edge_attrs)
                self._relationship_counts[relationship] += 1
                self.logger.info(f"Added relationship '{relationship}' between {source_id} and {target_id}.")

            return True

    # ==========================================
    # Query Methods (via Query)
    # ==========================================
    @eidosian()
    def get_node_id(self, term: Term) -> Optional[WordId]:
        """Retrieve node ID for a term. Delegates to GraphQuery."""
        return self.query.get_node_id(term)

    @eidosian()
    def get_related_terms(self, term: Term, rel_type: Optional[RelType] = None) -> List[Term]:
        """Find related terms. Delegates to GraphQuery."""
        return self.query.get_related_terms(term, rel_type)

    @eidosian()
    def get_node_count(self) -> int:
        """Get node count. Delegates to GraphQuery."""
        return self.query.get_node_count()

    @eidosian()
    def get_edge_count(self) -> int:
        """Get edge count. Delegates to GraphQuery."""
        return self.query.get_edge_count()

    @eidosian()
    def get_term_by_id(self, word_id: WordId) -> Optional[Term]:
        """Get term by ID. Delegates to GraphQuery."""
        return self.query.get_term_by_id(word_id)

    @eidosian()
    def get_graph_info(self) -> GraphInfoDict:
        """Get graph summary info. Delegates to GraphQuery."""
        return self.query.get_graph_info()

    @eidosian()
    def display_graph_summary(self) -> None:
        """Display graph summary. Delegates to GraphQuery."""
        self.query.display_graph_summary()

    @eidosian()
    def get_subgraph(self, term: Term, depth: int = 1) -> nx.Graph:
        """Extract a subgraph. Delegates to GraphQuery."""
        return self.query.get_subgraph(term, depth)

    @eidosian()
    def get_relationships_by_dimension(
        self,
        dimension: RelationshipDimension = "lexical",
        rel_type: Optional[RelType] = None,
        valence_range: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[Term, Term, RelType, Dict[str, Any]]]:
        """Get relationships filtered by dimension. Delegates to GraphQuery."""
        return self.query.get_relationships_by_dimension(dimension, rel_type, valence_range)

    # ==========================================
    # Layout Methods (via Layout)
    # ==========================================
    @eidosian()
    def compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """Compute graph layout. Delegates to GraphLayout."""
        # Layout computation can be read-heavy but writes to _positions
        # Lock ensures position dictionary isn't modified during read by visualizer
        with self._graph_lock:
            self.layout.compute_layout(algorithm)

    # update_layout_incrementally is called internally by add_word_node

    # ==========================================
    # Visualization Methods (via Visualizer)
    # ==========================================
    @eidosian()
    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate graph visualization. Delegates to GraphVisualizer."""
        # Visualization reads graph structure and positions, lock ensures consistency
        with self._graph_lock:
            self.visualizer.visualize(output_path, height, width, use_3d, dimensions_filter, open_in_browser)

    @eidosian()
    def visualize_2d(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate 2D graph visualization. Delegates to GraphVisualizer."""
        with self._graph_lock:
            self.visualizer.visualize_2d(output_path, height, width, dimensions_filter, open_in_browser)

    @eidosian()
    def visualize_3d(
        self,
        output_path: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """Generate 3D graph visualization. Delegates to GraphVisualizer."""
        with self._graph_lock:
            self.visualizer.visualize_3d(output_path, dimensions_filter, open_in_browser)

    # ==========================================
    # IO Methods (via IO)
    # ==========================================
    @eidosian()
    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """Save graph to GEXF. Delegates to GraphIO."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            self.io.save_to_gexf(path)

    @eidosian()
    def load_from_gexf(self, path: Optional[str] = None) -> None:
        """Load graph from GEXF. Delegates to GraphIO."""
        # Replaces graph structure, requires exclusive access
        with self._graph_lock:
            self.io.load_from_gexf(path)

    @eidosian()
    def export_subgraph(self, term: Term, depth: int = 1, output_path: Optional[str] = None) -> str:
        """Export subgraph to GEXF. Delegates to GraphIO."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            return self.io.export_subgraph(term, depth, output_path)

    # ==========================================
    # Analysis Methods (via Analysis)
    # ==========================================
    @eidosian()
    def analyze_semantic_clusters(
        self,
        min_community_size: int = 3,
        weight_attribute: Optional[str] = "weight",
        resolution: float = 1.0,
        random_state: Optional[int] = None,
    ) -> ClusterResult:
        """Analyze semantic clusters. Delegates to GraphAnalysis."""
        # Reads graph structure, lock ensures consistency
        with self._graph_lock:
            return self.analysis.analyze_semantic_clusters(
                min_community_size, weight_attribute, resolution, random_state
            )

    @eidosian()
    def analyze_multidimensional_relationships(self) -> MultiDimResult:
        """Analyze multidimensional relationships. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_multidimensional_relationships()

    @eidosian()
    def extract_meta_emotional_patterns(self) -> MetaEmotionalResult:
        """Extract meta-emotional patterns. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.extract_meta_emotional_patterns()

    @eidosian()
    def analyze_emotional_valence_distribution(
        self, dimension: RelationshipDimension = "emotional"
    ) -> ValenceDistResult:
        """Analyze emotional valence distribution. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_emotional_valence_distribution(dimension)

    @eidosian()
    def integrate_emotional_context(self, context_name: str, context_weights: Dict[str, float]) -> int:
        """Integrate emotional context. Delegates to GraphAnalysis."""
        # Potentially modifies graph state (_emotional_contexts), lock needed
        with self._graph_lock:
            return self.analysis.integrate_emotional_context(context_name, context_weights)

    @eidosian()
    def analyze_emotional_transitions(
        self, path_length: int = 2, min_transition_strength: float = 0.1
    ) -> TransitionResult:
        """Analyze emotional transitions. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.analyze_emotional_transitions(path_length, min_transition_strength)

    @eidosian()
    def get_emotional_subgraph(
        self,
        term: Term,
        depth: int = 1,
        context: Optional[Union[str, Dict[str, float]]] = None,
        emotional_types: Optional[List[RelType]] = None,
        min_intensity: float = 0.0,
    ) -> nx.Graph:
        """Get emotional subgraph. Delegates to GraphAnalysis."""
        with self._graph_lock:
            return self.analysis.get_emotional_subgraph(term, depth, context, emotional_types, min_intensity)

    # ==========================================
    # Internal Helper Methods
    # ==========================================
    def _get_relationship_properties(self, rel_type: RelType) -> RelationshipProperties:
        """Retrieve properties for a given relationship type."""
        # Access the global relationship_properties dictionary
        # Provide a default empty dict if type is unknown
        # Use .get with a default value for safety
        default_props = relationship_properties.get(
            "default", {"weight": 0.3, "color": "#aaaaaa", "bidirectional": True}
        )
        props = relationship_properties.get(rel_type.lower(), default_props)
        # Ensure type correctness using cast or TypedDict validation if needed
        # Make sure the returned dict conforms to RelationshipProperties structure
        return cast(
            RelationshipProperties,
            {
                "weight": props.get("weight", default_props["weight"]),
                "color": props.get("color", default_props["color"]),
                "bidirectional": props.get("bidirectional", default_props["bidirectional"]),
            },
        )

    def _determine_dimension(self, rel_type: RelType) -> RelationshipDimension:
        """Determine the primary dimension for a relationship type."""
        # Simple logic: Check known emotional/affective types first, else lexical
        # This could be made more sophisticated based on config or properties
        rel_type_lower = rel_type.lower()
        if rel_type_lower in self.config.emotional_relationship_colors:
            return "emotional"
        if rel_type_lower in self.config.affective_relationship_colors:
            return "affective"
        # Add checks for other dimensions (connotative, contextual) if defined
        # Default to lexical
        return "lexical"

    def __del__(self) -> None:
        """Ensure database connection is closed when manager is destroyed."""
        self.logger.info("GraphManager shutting down. Closing database connection.")
        self.db_manager.close()
