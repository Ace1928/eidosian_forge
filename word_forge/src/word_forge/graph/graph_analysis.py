"""
Contains complex graph analysis methods.

Encapsulates logic for advanced graph algorithms like community detection
(clustering), multidimensional analysis, emotional pattern extraction,
valence distribution, context integration, and transition analysis.
Adheres to Eidosian principles of modularity, precision, and analytical depth.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│  GraphAnalysis   │
    │ (Orchestrator)   │      │ (Complex Algos & │
    └────────┬─────────┘      │   Pattern Recog) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐      ┌──────────────────┐
    │    NetworkX      │      │ Optional Libs:   │
    │ (Graph Algos)    │      │ - community      │
    └──────────────────┘      │ - scipy/numpy    │
                              └──────────────────┘
"""

from __future__ import annotations
from eidosian_core import eidosian

import logging
import traceback
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx

# Optional dependencies for specific analyses
try:
    import community as community_louvain  # python-louvain package

    _community_louvain_available = True
except ImportError:
    _community_louvain_available = False
    community_louvain = None  # Define for type checking

try:
    import numpy as np

    _numpy_available = True
except ImportError:
    _numpy_available = False
    np = None  # Define for type checking

# Import necessary components
from word_forge.exceptions import GraphAnalysisError, NodeNotFoundError
from word_forge.graph.graph_config import RelationshipDimension, RelType, Term, WordId

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


# Type Aliases for complex return structures
ClusterNodeInfo = Dict[str, Union[str, float, List[str], bool, None]]
ClusterResult = Dict[int, List[ClusterNodeInfo]]
MultiDimResult = Dict[str, Any]  # Define more specifically if possible
MetaEmotionalResult = Dict[Term, List[Dict[str, Any]]]
ValenceDistResult = Dict[str, Any]  # Define more specifically if possible
TransitionResult = List[Dict[str, Any]]


class GraphAnalysis:
    """
    Performs advanced analysis on the knowledge graph structure and attributes.

    Includes methods for semantic clustering, multidimensional relationship
    analysis, emotional pattern extraction, valence distribution, context
    integration, and emotional transition pathway analysis. Requires optional
    dependencies (like python-louvain, numpy) for certain functionalities.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphAnalysis with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    @eidosian()
    def analyze_semantic_clusters(
        self,
        min_community_size: int = 3,
        weight_attribute: Optional[str] = "weight",  # Allow specifying weight attribute
        resolution: float = 1.0,
        random_state: Optional[int] = None,  # For reproducibility
    ) -> ClusterResult:
        """
        Identify semantic clusters (communities) within the graph using Louvain.

        Applies the Louvain community detection algorithm to partition the graph
        into clusters based on connection density. Filters out small communities.

        Args:
            min_community_size: Minimum number of nodes for a cluster to be included.
            weight_attribute: Edge attribute to use as weight for clustering.
                              Defaults to 'weight'. If None, uses unweighted graph.
            resolution: Louvain algorithm resolution parameter. Higher values lead
                        to more communities. Defaults to 1.0.
            random_state: Seed for the random number generator for reproducibility.

        Returns:
            ClusterResult: A dictionary where keys are cluster IDs (integers) and
                           values are lists of dictionaries, each describing a node
                           in the cluster (term, valence, etc.).

        Raises:
            GraphAnalysisError: If community detection fails or the required
                                'community' library is not installed.
        """
        if not _community_louvain_available:
            self.logger.error(
                "Community detection requires the 'python-louvain' library."
            )
            self.logger.error("Install with: pip install python-louvain")
            raise GraphAnalysisError(
                "Missing 'python-louvain' library for cluster analysis."
            )

        if self.manager.g.number_of_nodes() < min_community_size:
            self.logger.warning(
                f"Graph has fewer nodes ({self.manager.g.number_of_nodes()}) than min_community_size ({min_community_size}). Skipping clustering."
            )
            return {}

        self.logger.info(
            f"Analyzing semantic clusters (min size: {min_community_size}, resolution: {resolution})."
        )

        try:
            # Compute the best partition using the Louvain algorithm
            partition: Dict[WordId, int] = community_louvain.best_partition(
                self.manager.g,
                weight=weight_attribute,
                resolution=resolution,
                random_state=random_state,
            )

            # Group nodes by cluster ID
            clusters_raw: Dict[int, List[WordId]] = defaultdict(list)
            for node_id, cluster_id in partition.items():
                clusters_raw[cluster_id].append(node_id)

            # Format results and filter by size
            clusters_final: ClusterResult = {}
            for cluster_id, node_ids in clusters_raw.items():
                if len(node_ids) >= min_community_size:
                    cluster_nodes_info: List[ClusterNodeInfo] = []
                    for node_id in node_ids:
                        node_data = self.manager.g.nodes[node_id]
                        term = node_data.get("term", f"ID:{node_id}")
                        # Include relevant attributes like valence if available
                        node_info: ClusterNodeInfo = {
                            "id": node_id,
                            "term": term,
                            "valence": node_data.get("valence"),
                            "arousal": node_data.get("arousal"),
                            # Add other relevant node attributes as needed
                        }
                        cluster_nodes_info.append(node_info)
                    # Sort nodes within cluster alphabetically by term for consistency
                    clusters_final[cluster_id] = sorted(
                        cluster_nodes_info, key=lambda x: x.get("term", "")
                    )

            self.logger.info(
                f"Found {len(clusters_final)} clusters meeting minimum size criteria."
            )
            return clusters_final

        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphAnalysisError(
                f"Failed to analyze semantic clusters: {e}", e
            ) from e

    @eidosian()
    def analyze_multidimensional_relationships(self) -> MultiDimResult:
        """
        Analyze the distribution and interplay of relationships across dimensions.

        Calculates statistics about the prevalence of different relationship
        dimensions ('lexical', 'emotional', 'affective', etc.), identifies nodes
        participating in multiple dimensions, and finds the most common relationship
        types within each dimension.

        Returns:
            MultiDimResult: A dictionary containing analysis results:
                            - 'dimensions': Counts of edges per dimension.
                            - 'multi_dimensional_nodes': Nodes with edges in >1 dimension.
                            - 'most_common': Most frequent relationship type per dimension.
        """
        self.logger.info("Analyzing multidimensional relationship patterns.")
        dimension_counts: Dict[RelationshipDimension, int] = defaultdict(int)
        node_dimensions: Dict[WordId, Set[RelationshipDimension]] = defaultdict(set)
        type_counts_by_dimension: Dict[RelationshipDimension, Counter[RelType]] = (
            defaultdict(Counter)
        )

        # Iterate through edges to gather dimension and type data
        for u, v, data in self.manager.g.edges(data=True):
            dimension = data.get("dimension")
            rel_type = data.get("relationship")

            if dimension:
                # Ensure dimension is a valid RelationshipDimension type if possible
                # This might involve casting or validation depending on strictness
                valid_dimension = cast(RelationshipDimension, dimension)
                dimension_counts[valid_dimension] += 1
                node_dimensions[u].add(valid_dimension)
                node_dimensions[v].add(valid_dimension)
                if rel_type:
                    type_counts_by_dimension[valid_dimension][rel_type] += 1
            else:
                # Count edges without a dimension attribute if needed
                dimension_counts["unknown"] += 1  # type: ignore

        # Identify nodes participating in multiple dimensions
        multi_dimensional_nodes_info: Dict[Term, Dict[str, Any]] = {}
        for node_id, dims in node_dimensions.items():
            if len(dims) > 1:
                term = self.manager.query.get_term_by_id(node_id) or f"ID:{node_id}"
                multi_dimensional_nodes_info[term] = {
                    "id": node_id,
                    "dimensions": sorted(list(dims)),  # Store sorted list of dimensions
                }

        # Find the most common relationship type per dimension
        most_common_types: Dict[RelationshipDimension, List[Tuple[RelType, int]]] = {}
        for dimension, counts in type_counts_by_dimension.items():
            most_common_types[dimension] = counts.most_common(1)  # Get only the top one

        results: MultiDimResult = {
            "dimensions": dict(dimension_counts),
            "multi_dimensional_nodes": multi_dimensional_nodes_info,
            "most_common": most_common_types,
        }

        self.logger.info(
            f"Multidimensional analysis complete. Found relationships across {len(dimension_counts)} dimensions."
        )
        return results

    @eidosian()
    def extract_meta_emotional_patterns(self) -> MetaEmotionalResult:
        """
        Identify patterns where emotions relate to other emotions (meta-emotions).

        Searches for specific relationship types indicative of meta-emotional
        connections (e.g., 'meta_emotion', 'evokes', 'intensifies').

        Returns:
            MetaEmotionalResult: A dictionary where keys are source emotion terms
                                 and values are lists of target emotion info
                                 (term, relationship type).
        """
        self.logger.info("Extracting meta-emotional relationship patterns.")
        meta_patterns: MetaEmotionalResult = defaultdict(list)
        # Define relationship types considered 'meta-emotional'
        meta_rel_types: Set[RelType] = {
            "meta_emotion",
            "evokes",
            "responds_to",
            "intensifies",
            "diminishes",
            "emotional_component",  # Could be considered meta depending on definition
        }

        for u, v, data in self.manager.g.edges(data=True):
            rel_type = data.get("relationship")
            dimension = data.get("dimension")

            # Check if it's an emotional dimension and a meta-relationship type
            if dimension == "emotional" and rel_type in meta_rel_types:
                source_term = self.manager.query.get_term_by_id(u)
                target_term = self.manager.query.get_term_by_id(v)

                if source_term and target_term:
                    meta_patterns[source_term].append(
                        {
                            "term": target_term,
                            "relationship": rel_type,
                            "target_id": v,
                            "attributes": data,  # Include full edge data if needed
                        }
                    )

        self.logger.info(
            f"Found {len(meta_patterns)} source terms involved in meta-emotional patterns."
        )
        return dict(meta_patterns)  # Convert back to dict for return type consistency

    @eidosian()
    def analyze_emotional_valence_distribution(
        self, dimension: RelationshipDimension = "emotional"
    ) -> ValenceDistResult:
        """
        Analyze the distribution of emotional valence across nodes.

        Calculates statistics (mean, range) for node 'valence' attributes,
        identifies the most positive and negative terms based on valence.
        Focuses on nodes involved in the specified dimension's relationships.

        Args:
            dimension: The dimension to consider for node inclusion (default: 'emotional').
                       Nodes connected by edges of this dimension are analyzed.

        Returns:
            ValenceDistResult: A dictionary with statistics: 'count', 'mean',
                               'range' (min, max), 'top_positive', 'top_negative'.

        Raises:
            GraphAnalysisError: If NumPy is required but not available.
        """
        if not _numpy_available:
            self.logger.error("Valence distribution analysis requires NumPy.")
            self.logger.error("Install with: pip install numpy")
            raise GraphAnalysisError("Missing 'numpy' library for valence analysis.")

        self.logger.info(
            f"Analyzing emotional valence distribution for nodes in dimension '{dimension}'."
        )
        valences: List[float] = []
        node_valence_map: Dict[WordId, float] = {}

        # Identify nodes involved in the specified dimension
        relevant_nodes: Set[WordId] = set()
        for u, v, data in self.manager.g.edges(data=True):
            if data.get("dimension") == dimension:
                relevant_nodes.add(u)
                relevant_nodes.add(v)

        # Collect valences from relevant nodes
        for node_id in relevant_nodes:
            node_data = self.manager.g.nodes[node_id]
            valence = node_data.get("valence")
            if isinstance(valence, (int, float)):  # Check if valence is numeric
                val = float(valence)
                valences.append(val)
                node_valence_map[node_id] = val
            elif valence is not None:
                self.logger.warning(
                    f"Node {node_id} has non-numeric valence attribute '{valence}'. Skipping."
                )

        if not valences:
            self.logger.warning(
                f"No numeric valence data found for nodes in dimension '{dimension}'."
            )
            return {
                "count": 0,
                "mean": 0.0,
                "range": (0.0, 0.0),
                "top_positive": [],
                "top_negative": [],
            }

        # Calculate statistics using NumPy
        valence_array = np.array(valences)
        mean_valence = float(np.mean(valence_array))
        min_valence = float(np.min(valence_array))
        max_valence = float(np.max(valence_array))
        count = len(valences)

        # Find top positive and negative terms
        sorted_nodes = sorted(
            node_valence_map.items(), key=lambda item: item[1], reverse=True
        )
        top_positive_nodes = sorted_nodes[:5]
        top_negative_nodes = sorted_nodes[-5:][::-1]  # Get last 5 and reverse

        top_positive = [
            (self.manager.query.get_term_by_id(nid) or f"ID:{nid}", val)
            for nid, val in top_positive_nodes
        ]
        top_negative = [
            (self.manager.query.get_term_by_id(nid) or f"ID:{nid}", val)
            for nid, val in top_negative_nodes
        ]

        results: ValenceDistResult = {
            "count": count,
            "mean": mean_valence,
            "range": (min_valence, max_valence),
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

        self.logger.info(
            f"Valence analysis complete: {count} nodes, mean={mean_valence:.2f}, range=[{min_valence:.2f}, {max_valence:.2f}]."
        )
        return results

    @eidosian()
    def integrate_emotional_context(
        self, context_name: str, context_weights: Dict[str, float]
    ) -> int:
        """
        Apply an emotional context to the graph, potentially modifying edge weights.

        This is a placeholder for a more complex context integration logic.
        Currently, it stores the context but doesn't modify the graph structure.
        Future implementations could adjust edge weights or node attributes based
        on the context (e.g., 'professional' context might dampen 'anger' links).

        Args:
            context_name: A name for the emotional context (e.g., 'clinical').
            context_weights: A dictionary mapping emotional aspects (or relationship
                             types) to weighting factors (e.g., {"anger": 0.2}).

        Returns:
            int: The number of graph elements potentially affected (currently 0).
                 In a future implementation, this would be the count of modified
                 edges or nodes.
        """
        self.logger.info(f"Integrating emotional context '{context_name}'.")
        if not isinstance(context_weights, dict):
            raise TypeError("context_weights must be a dictionary.")

        # Store the context within the manager's state
        # Ensure the context name is valid (e.g., string, non-empty)
        if not context_name or not isinstance(context_name, str):
            raise ValueError("context_name must be a non-empty string.")

        self.manager._emotional_contexts[context_name] = context_weights
        self.logger.debug(
            f"Stored context '{context_name}' with weights: {context_weights}"
        )

        # Placeholder: Actual graph modification logic would go here.
        # Example: Iterate edges, check if dimension is emotional,
        # check if rel_type is in context_weights, adjust data['weight'].
        affected_count = 0
        # for u, v, data in self.manager.g.edges(data=True):
        #     if data.get("dimension") == "emotional":
        #         rel_type = data.get("relationship")
        #         if rel_type in context_weights:
        #             original_weight = data.get("weight", 1.0)
        #             modifier = context_weights[rel_type]
        #             data["weight"] = original_weight * modifier # Example modification
        #             data[f"weight_{context_name}"] = data["weight"] # Store context-specific weight
        #             affected_count += 1

        if affected_count > 0:
            self.logger.info(
                f"Applied context '{context_name}', modified {affected_count} emotional relationships."
            )
        else:
            self.logger.info(
                f"Context '{context_name}' stored. No graph modifications applied in current implementation."
            )

        # Return the number of affected elements (currently 0)
        return affected_count

    @eidosian()
    def analyze_emotional_transitions(
        self, path_length: int = 2, min_transition_strength: float = 0.1
    ) -> TransitionResult:
        """
        Analyze pathways of emotional transitions between terms.

        Identifies sequences of connected terms (paths) primarily linked by
        emotional relationships, calculating the overall strength and valence
        shift along the path.

        Args:
            path_length: The maximum length of transition paths to consider (number of edges).
            min_transition_strength: Minimum aggregated strength for a path to be included.

        Returns:
            TransitionResult: A list of dictionaries, each describing a transition path:
                              'path': List of terms in the sequence.
                              'strength': Aggregated strength (e.g., product of weights).
                              'valence_shift': Difference in valence between end and start nodes.
        """
        self.logger.info(
            f"Analyzing emotional transitions (max length: {path_length}, min strength: {min_transition_strength})."
        )
        transitions: TransitionResult = []

        # Consider only nodes involved in emotional relationships for efficiency
        emotional_nodes: Set[WordId] = set()
        for u, v, data in self.manager.g.edges(data=True):
            if data.get("dimension") == "emotional":
                emotional_nodes.add(u)
                emotional_nodes.add(v)

        if not emotional_nodes:
            self.logger.warning(
                "No nodes involved in emotional relationships found. Skipping transition analysis."
            )
            return []

        # Iterate through all pairs of emotional nodes as potential start/end points
        for start_node_id in emotional_nodes:
            # Find simple paths up to the specified length within the emotional subgraph view
            # Create a subgraph view containing only emotional edges for pathfinding
            emotional_edge_view = nx.subgraph_view(
                self.manager.g,
                filter_edge=lambda u, v: self.manager.g[u][v].get("dimension")
                == "emotional",
            )

            # Find paths starting from start_node_id
            # Limit path length by path_length + 1 (number of nodes)
            for target_node_id in emotional_nodes:
                if start_node_id == target_node_id:
                    continue

                # Use all_simple_paths for finding sequences
                # Note: This can be computationally expensive for large graphs/lengths
                try:
                    paths_generator = nx.all_simple_paths(
                        emotional_edge_view,
                        source=start_node_id,
                        target=target_node_id,
                        cutoff=path_length,
                    )

                    for path_node_ids in paths_generator:
                        if len(path_node_ids) < 2:  # Need at least start and end node
                            continue

                        path_terms: List[Term] = []
                        path_strength = 1.0
                        valid_path = True
                        for node_id in path_node_ids:
                            term = self.manager.query.get_term_by_id(node_id)
                            if term:
                                path_terms.append(term)
                            else:
                                valid_path = False
                                break  # Stop if any node lacks a term

                        if not valid_path:
                            continue

                        # Calculate path strength (e.g., product of edge weights)
                        for i in range(len(path_node_ids) - 1):
                            u, v = path_node_ids[i], path_node_ids[i + 1]
                            edge_data = self.manager.g.get_edge_data(u, v)
                            # Use default weight if missing
                            path_strength *= edge_data.get(
                                "weight", 0.1
                            )  # Use small default if missing

                        # Check against minimum strength threshold
                        if path_strength >= min_transition_strength:
                            # Calculate valence shift
                            start_valence = self.manager.g.nodes[start_node_id].get(
                                "valence", 0.0
                            )
                            end_valence = self.manager.g.nodes[target_node_id].get(
                                "valence", 0.0
                            )
                            # Ensure valences are numeric, default to 0.0 if not
                            start_valence = (
                                float(start_valence)
                                if isinstance(start_valence, (int, float))
                                else 0.0
                            )
                            end_valence = (
                                float(end_valence)
                                if isinstance(end_valence, (int, float))
                                else 0.0
                            )
                            valence_shift = end_valence - start_valence

                            transitions.append(
                                {
                                    "path": path_terms,
                                    "strength": path_strength,
                                    "valence_shift": valence_shift,
                                    "start_node_id": start_node_id,
                                    "end_node_id": target_node_id,
                                }
                            )

                except nx.NodeNotFound:
                    # This might happen if a node exists in emotional_nodes but not in the subgraph view (shouldn't typically)
                    self.logger.warning(
                        f"Node {start_node_id} or {target_node_id} not found in emotional subgraph view during pathfinding."
                    )
                    continue
                except Exception as path_err:
                    self.logger.error(
                        f"Error finding paths between {start_node_id} and {target_node_id}: {path_err}"
                    )
                    continue  # Continue to next pair

        # Sort transitions by strength (descending)
        transitions.sort(key=lambda x: x["strength"], reverse=True)

        self.logger.info(
            f"Found {len(transitions)} emotional transition pathways meeting criteria."
        )
        # Limit the number of returned transitions if necessary for performance
        # return transitions[:max_results]
        return transitions

    @eidosian()
    def get_emotional_subgraph(
        self,
        term: Term,
        depth: int = 1,
        context: Optional[Union[str, Dict[str, float]]] = None,
        emotional_types: Optional[List[RelType]] = None,
        min_intensity: float = 0.0,  # Example filter: minimum edge weight/intensity
    ) -> nx.Graph:
        """
        Extract a subgraph focusing on emotional relationships around a term.

        Filters the subgraph based on emotional criteria like relationship types,
        contextual weights (future), and minimum intensity/weight.

        Args:
            term: The central term for the subgraph.
            depth: The maximum distance from the central term.
            context: Optional emotional context name (str) or weights (dict) to apply.
                     (Currently placeholder - does not modify graph).
            emotional_types: Optional list of specific emotional relationship
                             types to include. If None, includes all emotional edges.
            min_intensity: Minimum weight for an emotional edge to be included.

        Returns:
            nx.Graph: A NetworkX graph object representing the emotional subgraph.

        Raises:
            NodeNotFoundError: If the input term is not found.
            ValueError: If depth is negative.
        """
        if depth < 0:
            raise ValueError("Subgraph depth cannot be negative.")

        start_node_id = self.manager.query.get_node_id(term)
        if start_node_id is None:
            raise NodeNotFoundError(
                f"Term '{term}' not found for emotional subgraph extraction."
            )

        self.logger.info(
            f"Extracting emotional subgraph for '{term}' (depth: {depth}, types: {emotional_types}, min_intensity: {min_intensity}, context: {context})."
        )

        # 1. Get the initial neighborhood subgraph based on depth
        neighborhood_nodes = nx.ego_graph(
            self.manager.g, start_node_id, radius=depth
        ).nodes()
        base_subgraph = self.manager.g.subgraph(neighborhood_nodes)

        # 2. Filter edges based on emotional criteria
        emotional_subgraph = (
            nx.Graph()
        )  # Create a new graph to add filtered nodes/edges
        for node, data in base_subgraph.nodes(data=True):
            emotional_subgraph.add_node(node, **data)  # Copy nodes

        context_weights_local: Optional[Dict[str, float]] = None
        if context:
            if isinstance(context, str):
                context_weights_local = self.manager._emotional_contexts.get(
                    context, {}
                )
                if not context_weights_local:
                    self.logger.warning(
                        f"Context '{context}' not found. Proceeding without weights."
                    )
            elif isinstance(context, dict):
                context_weights_local = context
            else:
                raise TypeError("context must be a string or a dictionary of weights")

        for u, v, data in base_subgraph.edges(data=True):
            is_emotional = data.get("dimension") == "emotional"
            if not is_emotional:
                continue  # Skip non-emotional edges

            # Filter by emotional type if specified
            rel_type = data.get("relationship")
            if emotional_types is not None and rel_type not in emotional_types:
                continue

            # Determine weight, possibly modified by context
            weight = data.get("weight", 0.0)
            if context_weights_local:
                modifier = context_weights_local.get(
                    rel_type, context_weights_local.get("default", 1.0)
                )
                weight *= modifier
                data = dict(data)
                data["weight"] = weight

            # Filter by minimum intensity (after applying context)
            if weight < min_intensity:
                continue

            # If all filters pass, add the edge
            emotional_subgraph.add_edge(u, v, **data)

        num_nodes = emotional_subgraph.number_of_nodes()
        num_edges = emotional_subgraph.number_of_edges()
        self.logger.info(
            f"Emotional subgraph extracted: {num_nodes} nodes, {num_edges} edges."
        )

        # Remove isolated nodes (nodes that were in the neighborhood but have no *emotional* edges left after filtering)
        isolated = [node for node, degree in emotional_subgraph.degree() if degree == 0]
        if isolated:
            if hasattr(emotional_subgraph, "remove_nodes_from"):
                emotional_subgraph.remove_nodes_from(isolated)
            else:
                for node in isolated:
                    if node in emotional_subgraph._nodes:
                        del emotional_subgraph._nodes[node]
                emotional_subgraph._edges = {
                    (u, v): d
                    for (u, v), d in emotional_subgraph._edges.items()
                    if u not in isolated and v not in isolated
                }
            self.logger.debug(
                f"Removed {len(isolated)} isolated nodes from emotional subgraph."
            )

        # Ensure returned graph has iterable 'nodes' attribute for stubs
        if not isinstance(getattr(emotional_subgraph, "nodes"), list):
            try:
                emotional_subgraph.nodes = list(emotional_subgraph.nodes())
            except Exception:
                pass

        return emotional_subgraph
