"""
Manages the generation of interactive graph visualizations.

Encapsulates logic for creating 2D and 3D visualizations using libraries
like Pyvis and Plotly. Handles configuration of visual elements (nodes, edges,
layout) based on GraphConfig settings. Adheres to Eidosian principles of
modularity, clarity, and effective communication.

Architecture:
    ┌──────────────────┐      ┌────────────────────┐
    │  GraphManager    │◄────►│  GraphVisualizer   │
    │ (Orchestrator)   │      │ (Plotting & Config)│
    └────────┬─────────┘      └─────────┬──────────┘
             │                          │
             ▼                          ▼
    ┌──────────────────┐      ┌────────────────────┐
    │    GraphLayout   │      │ Visualization Libs │
    │  (Positions)     │      │ (Pyvis, Plotly)    │
    └──────────────────┘      └────────────────────┘
"""

from __future__ import annotations

import json
import logging
import traceback
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional

import networkx as nx

# Optional dependencies for visualization
try:
    from pyvis.network import Network as PyvisNetwork

    _pyvis_available = True
except ImportError:
    _pyvis_available = False
    PyvisNetwork = None  # Define for type checking

try:
    import plotly.graph_objects as go

    _plotly_available = True
except ImportError:
    _plotly_available = False
    go = None  # Define for type checking

_VISUALIZATION_INSTALL_HINT = 'pip install "word_forge[visualization]"'

# Emotional dimension thresholds for color classification
# These can be adjusted to tune sensitivity of emotional coloring
VALENCE_HIGH_THRESHOLD: Final[float] = 0.3  # Above this = positive
VALENCE_LOW_THRESHOLD: Final[float] = -0.3  # Below this = negative
AROUSAL_HIGH_THRESHOLD: Final[float] = 0.5  # Above this = high arousal

# Import necessary components
from word_forge.exceptions import GraphVisualizationError
from word_forge.graph.graph_config import (
    ColorHex,
    Position,
    PositionDict,  # Ensure PositionDict is imported
    RelationshipDimension,
    WordId,
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphVisualizer:
    """
    Generates interactive 2D and 3D visualizations of the knowledge graph.

    Uses Pyvis for 2D visualizations and Plotly for 3D visualizations.
    Configures node size, color, edge properties, and layout based on the
    GraphManager's state and configuration.

    Attributes:
        manager (GraphManager): Reference to the main GraphManager for state access.
        logger (logging.Logger): Logger instance for this module.
        _config (GraphConfig): Reference to the graph configuration object.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphVisualizer with a reference to the GraphManager.

        Args:
            manager (GraphManager): The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    def visualize(
        self,
        output_path: Optional[str] = None,
        height: Optional[
            str
        ] = None,  # Pyvis expects string height/width (e.g., "800px")
        width: Optional[str] = None,
        use_3d: Optional[bool] = None,  # Explicitly choose 2D/3D
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,  # Option to automatically open
    ) -> None:
        """
        Generate an interactive graph visualization (2D default, optionally 3D).

        Creates an HTML file containing the visualization. Uses Pyvis for 2D
        and Plotly for 3D. Filters graph elements based on provided dimensions.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            height (Optional[str]): Height of the visualization canvas (e.g., "800px"). Defaults to config.
            width (Optional[str]): Width of the visualization canvas (e.g., "1200px"). Defaults to config.
            use_3d (Optional[bool]): If True, generate a 3D plot using Plotly. If False or None,
                                     generate a 2D plot using Pyvis (respecting manager.dimensions if 3D).
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.

        Raises:
            GraphVisualizationError: If visualization libraries are missing or
                                     if generation fails.
            GraphError: If the graph is empty or positions are missing.
        """
        is_3d = use_3d if use_3d is not None else (self.manager.dimensions == 3)

        if is_3d:
            self.visualize_3d(output_path, dimensions_filter, open_in_browser)
        else:
            self.visualize_2d(
                output_path, height, width, dimensions_filter, open_in_browser
            )

    def visualize_2d(
        self,
        output_path: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """
        Generate an interactive 2D graph visualization using Pyvis.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            height (Optional[str]): Height of the visualization canvas (e.g., "800px"). Defaults to config.
            width (Optional[str]): Width of the visualization canvas (e.g., "1200px"). Defaults to config.
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.

        Raises:
            GraphVisualizationError: If Pyvis is not installed or generation fails.
            GraphError: If the graph is empty or 2D positions are missing.
        """
        if not _pyvis_available:
            self.logger.error("Pyvis library is required for 2D visualization.")
            self.logger.error(
                f"Install with the visualization extras: {_VISUALIZATION_INSTALL_HINT}"
            )
            raise GraphVisualizationError(
                "Missing 'pyvis' library. Install the visualization extras to enable 2D graph rendering."
            )

        if self.manager.g.number_of_nodes() == 0:
            raise GraphVisualizationError("Cannot visualize an empty graph.")

        node_positions = self.manager.get_positions()
        if not node_positions:
            self.logger.warning(
                "Node positions not computed. Computing default layout for 2D visualization."
            )
            try:
                original_dims = self.manager.dimensions
                if original_dims != 2:
                    self.manager.dimensions = 2
                self.manager.layout.compute_layout()
                node_positions = self.manager.get_positions()  # Re-fetch positions
                if original_dims != 2:
                    self.manager.dimensions = original_dims
            except Exception as e:
                raise GraphVisualizationError(
                    "Failed to compute layout for visualization.", e
                ) from e

        sample_pos: Optional[Position] = next(iter(node_positions.values()), None)
        if sample_pos is not None and len(sample_pos) != 2:
            self.logger.warning(
                "Graph positions are 3D, but 2D visualization requested. Using only X, Y coordinates."
            )

        vis_height = height or f"{self._config.vis_height}px"
        vis_width = width or f"{self._config.vis_width}px"
        save_path_str = output_path or self._config.visualization_path
        save_path = Path(save_path_str)

        if save_path.is_dir() or save_path.suffix.lower() != ".html":
            default_filename = "graph_2d.html"
            save_path = (
                save_path / default_filename
                if save_path.is_dir()
                else save_path.with_name(save_path.stem + "_2d.html")
            )
            self.logger.debug(f"Adjusted 2D visualization save path to: {save_path}")

        self.logger.info(f"Generating 2D visualization (Pyvis) to: {save_path}")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GraphVisualizationError(
                f"Could not create directory for visualization: {save_path.parent}", e
            ) from e

        net = PyvisNetwork(
            height=vis_height,
            width=vis_width,
            directed=isinstance(self.manager.g, nx.DiGraph),
            notebook=False,
            bgcolor="#222222",
            font_color="white",
        )

        graph_to_visualize = self._filter_graph_by_dimensions(dimensions_filter)

        # Configure Pyvis network appearance and add nodes/edges
        self._configure_pyvis_network(net, graph_to_visualize, node_positions)

        try:
            net.save_graph(str(save_path))
            self.logger.info(f"2D visualization saved successfully to {save_path}")

            if open_in_browser:
                try:
                    webbrowser.open(f"file://{str(save_path.resolve())}")
                except Exception as wb_err:
                    self.logger.warning(
                        f"Could not automatically open visualization in browser: {wb_err}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to generate or save Pyvis visualization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphVisualizationError(
                f"Error generating 2D visualization: {e}", e
            ) from e

    def visualize_3d(
        self,
        output_path: Optional[str] = None,
        dimensions_filter: Optional[List[RelationshipDimension]] = None,
        open_in_browser: bool = False,
    ) -> None:
        """
        Generate an interactive 3D graph visualization using Plotly.

        Args:
            output_path (Optional[str]): Path to save the HTML file. Defaults to config path.
            dimensions_filter (Optional[List[RelationshipDimension]]): List of relationship dimensions to include.
                                                                       If None, includes dimensions specified in config.active_dimensions.
            open_in_browser (bool): If True, automatically opens the generated HTML file.

        Raises:
            GraphVisualizationError: If Plotly is not installed or generation fails.
            GraphError: If the graph is empty or 3D positions are missing.
        """
        if not _plotly_available:
            self.logger.error("Plotly library is required for 3D visualization.")
            self.logger.error(
                f"Install with the visualization extras: {_VISUALIZATION_INSTALL_HINT}"
            )
            raise GraphVisualizationError(
                "Missing 'plotly' library. Install the visualization extras to enable 3D graph rendering."
            )

        if self.manager.g.number_of_nodes() == 0:
            raise GraphVisualizationError("Cannot visualize an empty graph.")

        node_positions = self.manager.get_positions()
        if not node_positions:
            self.logger.warning(
                "Node positions not computed. Computing default 3D layout."
            )
            try:
                original_dims = self.manager.dimensions
                if original_dims != 3:
                    self.manager.dimensions = 3
                self.manager.layout.compute_layout()
                node_positions = self.manager.get_positions()  # Re-fetch
                if original_dims != 3:
                    self.manager.dimensions = original_dims
            except Exception as e:
                raise GraphVisualizationError(
                    "Failed to compute 3D layout for visualization.", e
                ) from e

        sample_pos: Optional[Position] = next(iter(node_positions.values()), None)
        if sample_pos is not None and len(sample_pos) != 3:
            self.logger.warning(
                "Graph positions are 2D, but 3D visualization requested. Computing 3D layout."
            )
            try:
                self.manager.dimensions = 3
                self.manager.layout.compute_layout()
                node_positions = self.manager.get_positions()  # Re-fetch
            except Exception as e:
                raise GraphVisualizationError(
                    "Failed to compute 3D layout for visualization.", e
                ) from e

        save_path_str = output_path or self._config.visualization_path
        save_path = Path(save_path_str)

        if save_path.is_dir() or save_path.suffix.lower() != ".html":
            default_filename = "graph_3d.html"
            save_path = (
                save_path / default_filename
                if save_path.is_dir()
                else save_path.with_name(save_path.stem + "_3d.html")
            )
            self.logger.debug(f"Adjusted 3D visualization save path to: {save_path}")

        self.logger.info(f"Generating 3D visualization (Plotly) to: {save_path}")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GraphVisualizationError(
                f"Could not create directory for visualization: {save_path.parent}", e
            ) from e

        graph_to_visualize = self._filter_graph_by_dimensions(dimensions_filter)

        # Create Plotly figure using the potentially updated positions
        fig = self._configure_plotly_figure(graph_to_visualize, node_positions)

        try:
            fig.write_html(str(save_path))
            self.logger.info(f"3D visualization saved successfully to {save_path}")

            if open_in_browser:
                try:
                    webbrowser.open(f"file://{str(save_path.resolve())}")
                except Exception as wb_err:
                    self.logger.warning(
                        f"Could not automatically open visualization in browser: {wb_err}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to generate or save Plotly visualization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphVisualizationError(
                f"Error generating 3D visualization: {e}", e
            ) from e

    def _filter_graph_by_dimensions(
        self, dimensions_filter: Optional[List[RelationshipDimension]]
    ) -> nx.Graph:
        """
        Create a subgraph containing only edges matching the dimension filter.

        Args:
            dimensions_filter (Optional[List[RelationshipDimension]]): List of dimensions to include.
                                                                       If None, uses config.active_dimensions.

        Returns:
            nx.Graph: A NetworkX graph object containing the filtered edges and involved nodes.
                      Returns a copy to prevent modification of the original graph view.
        """
        active_dimensions = set(dimensions_filter or self._config.active_dimensions)
        self.logger.debug(f"Filtering graph for dimensions: {active_dimensions}")

        if not active_dimensions:
            self.logger.warning(
                "Dimension filter is empty. Visualization might be empty."
            )
            return type(self.manager.g)()

        def edge_filter(u: WordId, v: WordId) -> bool:
            """Check if edge between u and v matches active dimensions."""
            if self.manager.g.is_multigraph():
                if not self.manager.g.has_edge(u, v):
                    return False
                for key in self.manager.g[u][v]:
                    edge_data = self.manager.g.get_edge_data(u, v, key=key)
                    if edge_data and edge_data.get("dimension") in active_dimensions:
                        return True
                return False
            else:
                edge_data = self.manager.g.get_edge_data(u, v)
                return (
                    edge_data is not None
                    and edge_data.get("dimension") in active_dimensions
                )

        filtered_view = nx.subgraph_view(self.manager.g, filter_edge=edge_filter)
        filtered_graph = type(self.manager.g)()
        filtered_graph.add_nodes_from(filtered_view.nodes(data=True))
        filtered_graph.add_edges_from(filtered_view.edges(data=True))
        return filtered_graph

    def _configure_pyvis_network(
        self, net: PyvisNetwork, graph: nx.Graph, node_positions: PositionDict
    ) -> None:
        """
        Configure Pyvis network object with nodes, edges, and styling.

        Args:
            net (PyvisNetwork): The PyvisNetwork instance.
            graph (nx.Graph): The NetworkX graph to visualize (potentially filtered).
            node_positions (PositionDict): Dictionary mapping node IDs to positions.
        """
        if not _pyvis_available:
            self.logger.error("Pyvis not available, cannot configure network.")
            return

        self.logger.debug("Configuring Pyvis network...")

        # Add nodes with attributes
        for node_id, attrs in graph.nodes(data=True):
            term = attrs.get("term", f"ID:{node_id}")
            node_size = self._calculate_node_size(node_id, graph)
            node_color = self._get_node_color(attrs)
            pos = node_positions.get(node_id)

            pos_x = float(pos[0] * 100) if pos is not None and len(pos) >= 1 else 0.0
            pos_y = float(pos[1] * 100) if pos is not None and len(pos) >= 2 else 0.0

            title_parts = [f"Term: {term}", f"ID: {node_id}"]
            if "valence" in attrs and attrs["valence"] is not None:
                title_parts.append(f"Valence: {attrs['valence']:.2f}")
            if "arousal" in attrs and attrs["arousal"] is not None:
                title_parts.append(f"Arousal: {attrs['arousal']:.2f}")
            title = "\n".join(title_parts) if self._config.enable_tooltips else None

            net.add_node(
                str(node_id),
                label=term if self._config.enable_labels else "",
                title=title,
                size=node_size,
                color=node_color,
                x=pos_x,
                y=pos_y,
                physics=False,
            )

        # Add edges with attributes
        for u, v, attrs in graph.edges(data=True):
            rel_type = attrs.get("relationship", "")
            dimension = attrs.get("dimension", "lexical")

            # Get color based on relationship type, falling back to dimension-based color
            edge_color = self._get_edge_color(rel_type, dimension, attrs)
            edge_width = self._calculate_edge_width(attrs.get("weight", 1.0))

            # Build informative tooltip
            title_parts = []
            if rel_type:
                title_parts.append(f"Type: {rel_type}")
            title_parts.append(f"Dimension: {dimension}")
            if attrs.get("valence") is not None:
                title_parts.append(f"Valence: {attrs['valence']:.2f}")
            if attrs.get("arousal") is not None:
                title_parts.append(f"Arousal: {attrs['arousal']:.2f}")
            if attrs.get("weight") is not None:
                title_parts.append(f"Weight: {attrs['weight']:.2f}")
            title = "\n".join(title_parts) if self._config.enable_tooltips else None

            # Use dashed style for cross-dimensional edges or emotional relationships
            style = attrs.get("style", "solid")
            if dimension in ("emotional", "affective") and style == "solid":
                style = self._config.cross_dimension_edge_style

            net.add_edge(
                str(u),
                str(v),
                title=title,
                color=edge_color,
                width=edge_width,
                label=rel_type if self._config.enable_edge_labels else "",
                dashes=(style == "dashed"),
            )

        # Apply general Pyvis options using set_options
        options_dict = {
            "physics": {"enabled": False},  # Disable physics for precomputed layout
            "layout": {
                "hierarchical": {
                    "enabled": self._config.default_layout
                    == "hierarchical"
                    # Add other hierarchical options if needed
                }
            },
            "nodes": {
                "font": {"color": "white"},
                "shape": "dot",  # Default shape
                # Add other global node options
            },
            "edges": {
                "smooth": {"enabled": True, "type": "continuous"},
                "font": {"color": "white", "size": 10, "align": "top"},
                # Add other global edge options
            },
            "interaction": {
                "tooltipDelay": 200,
                "navigationButtons": True,
                "keyboard": True,
            },
        }
        # Convert dict to JSON string for set_options
        options_json = json.dumps(options_dict)
        net.set_options(options_json)

        self.logger.debug("Pyvis network configuration complete.")

    def _configure_plotly_figure(
        self, graph: nx.Graph, node_positions: PositionDict
    ) -> go.Figure:
        """
        Configure Plotly figure object for 3D visualization.

        Args:
            graph (nx.Graph): The NetworkX graph to visualize (potentially filtered).
            node_positions (PositionDict): Dictionary mapping node IDs to positions.

        Returns:
            go.Figure: A Plotly Figure object.

        Raises:
            GraphVisualizationError: If Plotly is not available or no valid nodes found.
        """
        if not _plotly_available:
            raise GraphVisualizationError(
                "Plotly library missing. Install the visualization extras to configure 3D figures."
            )

        self.logger.debug("Configuring Plotly 3D figure...")

        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        edge_z: List[Optional[float]] = []

        for edge in graph.edges(data=True):
            u, v, data = edge
            pos_u = node_positions.get(u)
            pos_v = node_positions.get(v)

            # Explicitly check if positions are not None and have length 3
            if (
                pos_u is not None
                and pos_v is not None
                and len(pos_u) == 3
                and len(pos_v) == 3
            ):
                edge_x.extend([pos_u[0], pos_v[0], None])
                edge_y.extend([pos_u[1], pos_v[1], None])
                edge_z.extend([pos_u[2], pos_v[2], None])
            else:
                self.logger.warning(
                    f"Skipping edge ({u},{v}) due to missing or invalid 3D positions (pos_u: {type(pos_u)}, pos_v: {type(pos_v)})."
                )

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=self._config.min_edge_width, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x: List[float] = []
        node_y: List[float] = []
        node_z: List[float] = []
        node_text: List[str] = []
        node_sizes: List[float] = []
        node_colors: List[ColorHex] = []
        valid_node_ids = []

        for node_id, attrs in graph.nodes(data=True):
            pos = node_positions.get(node_id)
            # Explicitly check if position is not None and has length 3
            if pos is not None and len(pos) == 3:
                valid_node_ids.append(node_id)
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_z.append(pos[2])

                term = attrs.get("term", f"ID:{node_id}")
                hover_parts = [f"Term: {term}", f"ID: {node_id}"]
                if "valence" in attrs and attrs["valence"] is not None:
                    hover_parts.append(f"Valence: {attrs['valence']:.2f}")
                if "arousal" in attrs and attrs["arousal"] is not None:
                    hover_parts.append(f"Arousal: {attrs['arousal']:.2f}")
                node_text.append("<br>".join(hover_parts))

                node_sizes.append(self._calculate_node_size(node_id, graph))
                node_colors.append(self._get_node_color(attrs))
            else:
                self.logger.warning(
                    f"Skipping node {node_id} due to missing or invalid 3D position (pos: {type(pos)})."
                )

        if not valid_node_ids:
            self.logger.error(
                "No nodes with valid 3D positions found. Cannot generate 3D plot."
            )
            raise GraphVisualizationError(
                "No nodes with valid 3D positions found for Plotly."
            )

        # Calculate sizeref based on valid node sizes
        valid_node_sizes = [s for s in node_sizes if s is not None]
        sizeref_value = (
            (max(valid_node_sizes) / (self._config.max_node_size * 1.5))
            if valid_node_sizes
            else 1
        )

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers" + ("+text" if self._config.enable_labels else ""),
            hoverinfo="text" if self._config.enable_tooltips else "none",
            text=node_text if self._config.enable_tooltips else None,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                sizeref=sizeref_value,  # Use calculated sizeref
                sizemin=self._config.min_node_size / 1.5,
                line_width=0.5,
            ),
            textfont=(
                dict(size=10, color="#CCCCCC") if self._config.enable_labels else None
            ),
            textposition="top center",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                # Corrected: Use 'title' dict with 'text' and 'font' sub-properties
                title=dict(
                    text="<br>3D Knowledge Graph Visualization",
                    font=dict(size=16, color="white"),
                    x=0.5,  # Center title
                    xanchor="center",
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    zaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False, title=""
                    ),
                    bgcolor="#111111",
                ),
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#1e1e1e",
                font=dict(color="white"),  # Global font color
            ),
        )

        self.logger.debug("Plotly 3D figure configuration complete.")
        return fig

    def _calculate_node_size(self, node_id: WordId, graph: nx.Graph) -> float:
        """
        Calculate node size based on configured strategy.

        Args:
            node_id (WordId): The ID of the node.
            graph (nx.Graph): The graph containing the node (used for degree calculation).

        Returns:
            float: The calculated node size, clamped within configured min/max bounds.
        """
        strategy = self._config.node_size_strategy
        min_size = self._config.min_node_size
        max_size = self._config.max_node_size
        default_size = (min_size + max_size) / 2.0

        try:
            if strategy == "degree":
                if node_id not in graph:
                    return default_size
                degree = graph.degree(node_id)
                all_degrees = [d for n, d in graph.degree()]
                max_degree = max(all_degrees) if all_degrees else 1
                size = min_size + (max_size - min_size) * (degree / max(1, max_degree))
                return max(min_size, min(size, max_size))
            elif strategy == "centrality":
                self.logger.warning(
                    "Node size strategy 'centrality' not fully implemented, using 'degree'."
                )
                if node_id not in graph:
                    return default_size
                degree = graph.degree(node_id)
                all_degrees = [d for n, d in graph.degree()]
                max_degree = max(all_degrees) if all_degrees else 1
                size = min_size + (max_size - min_size) * (degree / max(1, max_degree))
                return max(min_size, min(size, max_size))
            else:
                return default_size
        except Exception as e:
            self.logger.warning(
                f"Error calculating node size for {node_id} using strategy '{strategy}': {e}. Using default size {default_size}."
            )
            return default_size

    def _calculate_edge_width(self, weight: Optional[float]) -> float:
        """
        Calculate edge width based on weight, clamped within configured bounds.

        Args:
            weight (Optional[float]): The edge weight (typically 0.0 to 1.0). Defaults to 0.5 if None.

        Returns:
            float: The calculated edge width.
        """
        min_width = self._config.min_edge_width
        max_width = self._config.max_edge_width
        effective_weight = weight if weight is not None else 0.5
        width = min_width + (max_width - min_width) * effective_weight
        return max(min_width, min(width, max_width))

    def _get_node_color(self, node_attributes: Dict[str, Any]) -> ColorHex:
        """
        Determine node color based on attributes and configuration.

        Implements coloring based on valence if available and configured,
        otherwise uses a default color.

        Args:
            node_attributes (Dict[str, Any]): Dictionary of attributes for the node.

        Returns:
            ColorHex: A hex color string (e.g., "#RRGGBB").
        """
        valence = node_attributes.get("valence")
        if isinstance(valence, (int, float)):
            if valence > 0.5:
                return self._config.affective_relationship_colors.get(
                    "positive_valence", "#00cc66"
                )
            if valence < -0.5:
                return self._config.affective_relationship_colors.get(
                    "negative_valence", "#cc3300"
                )
            if valence > 0.1:
                return "#90EE90"
            if valence < -0.1:
                return "#FFA07A"
            return self._config.affective_relationship_colors.get(
                "valence_neutral", "#cccccc"
            )

        return "#6666ff"  # Default blueish color

    def _get_edge_color(
        self,
        rel_type: str,
        dimension: str,
        attrs: Dict[str, Any],
    ) -> ColorHex:
        """
        Determine edge color based on relationship type, dimension, and attributes.

        Prioritizes relationship type colors, then dimension-based colors,
        and finally considers emotional attributes like valence for coloring.

        Args:
            rel_type: The relationship type (e.g., 'synonym', 'emotional_synonym')
            dimension: The relationship dimension (e.g., 'lexical', 'emotional')
            attrs: Edge attributes dictionary

        Returns:
            ColorHex: A hex color string for the edge.
        """
        # First, try to get color by relationship type if it's a known type
        if rel_type and rel_type.lower() in self._config.relationship_colors:
            return self._config.relationship_colors[rel_type.lower()]

        # Check emotional relationship colors
        if rel_type and rel_type.lower() in self._config.emotional_relationship_colors:
            return self._config.emotional_relationship_colors[rel_type.lower()]

        # Check affective relationship colors
        if rel_type and rel_type.lower() in self._config.affective_relationship_colors:
            return self._config.affective_relationship_colors[rel_type.lower()]

        # For emotional dimensions, color by valence if available
        if dimension in ("emotional", "affective"):
            valence = attrs.get("valence")
            if isinstance(valence, (int, float)):
                if valence > VALENCE_HIGH_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "positive_valence", "#00cc66"
                    )
                elif valence < VALENCE_LOW_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "negative_valence", "#cc3300"
                    )
                else:
                    return self._config.affective_relationship_colors.get(
                        "valence_neutral", "#cccccc"
                    )

            # Color by arousal if valence not available
            arousal = attrs.get("arousal")
            if isinstance(arousal, (int, float)):
                if arousal > AROUSAL_HIGH_THRESHOLD:
                    return self._config.affective_relationship_colors.get(
                        "high_arousal", "#ff9900"
                    )
                else:
                    return self._config.affective_relationship_colors.get(
                        "low_arousal", "#3366cc"
                    )

        # Dimension-based default colors
        dimension_colors = {
            "lexical": "#4287f5",  # Blue
            "emotional": "#ff69b4",  # Hot pink
            "affective": "#ff9900",  # Orange
            "contextual": "#20b2aa",  # Light sea green
            "connotative": "#daa520",  # Goldenrod
        }

        return dimension_colors.get(dimension, "#aaaaaa")
