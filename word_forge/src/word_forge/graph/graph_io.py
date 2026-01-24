"""
Handles graph input/output operations, such as saving and loading.

Encapsulates logic for serializing the graph to various formats (GEXF, etc.)
and deserializing graph data from files. Includes functionality for exporting
subgraphs. Adheres to Eidosian principles of modularity, robustness, and clarity.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│     GraphIO      │
    │ (Orchestrator)   │      │ (Serialization & │
    └────────┬─────────┘      │   File Handling) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐      ┌──────────────────┐
    │    NetworkX      │      │      File        │
    │ (Graph I/O Funcs)│      │     System       │
    └──────────────────┘      └──────────────────┘
"""

from __future__ import annotations
from eidosian_core import eidosian

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import networkx as nx

# Import necessary components
from word_forge.exceptions import GraphIOError, NodeNotFoundError
from word_forge.graph.graph_config import Term

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphIO:
    """
    Manages saving, loading, and exporting graph data.

    Provides methods to serialize the current graph state to files (e.g., GEXF)
    and load graph data from such files. Also supports exporting specific
    subgraphs based on a starting node and depth.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphIO with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    @eidosian()
    def save_to_gexf(self, path: Optional[str] = None) -> None:
        """
        Save the current graph state to a GEXF file.

        Serializes the graph managed by the GraphManager into the GEXF format,
        which preserves node and edge attributes suitable for visualization tools
        like Gephi.

        Args:
            path: The file path to save the GEXF file. If None, uses the
                  default path from the GraphConfig.

        Raises:
            GraphIOError: If the graph is empty or saving fails due to I/O issues
                          or NetworkX errors.
        """
        if self.manager.g.number_of_nodes() == 0:
            self.logger.warning("Attempted to save an empty graph. Operation skipped.")
            # Consider raising an error or just returning based on desired behavior
            # raise GraphIOError("Cannot save an empty graph.")
            return

        save_path_str = path or self._config.default_export_path
        save_path = Path(save_path_str)

        # Ensure the target directory exists
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {save_path.parent}")
        except OSError as e:
            self.logger.error(f"Failed to create directory {save_path.parent}: {e}")
            raise GraphIOError(
                f"Could not create directory for GEXF file: {save_path.parent}", e
            ) from e

        # Ensure the path includes the .gexf extension
        if save_path.suffix.lower() != ".gexf":
            save_path = save_path.with_suffix(".gexf")
            self.logger.debug(
                f"Adjusted save path to include .gexf extension: {save_path}"
            )

        self.logger.info(
            f"Saving graph ({self.manager.g.number_of_nodes()} nodes, {self.manager.g.number_of_edges()} edges) to GEXF: {save_path}"
        )

        try:
            # Prepare graph for GEXF: Convert complex attributes if necessary
            # GEXF standard requires specific types (string, integer, float, boolean)
            # NetworkX attempts conversion, but explicit handling might be needed
            # for custom objects or complex types stored as attributes.
            # Example (if positions were tuples/lists):
            # g_copy = self.manager.g.copy()
            # for node, data in g_copy.nodes(data=True):
            #     if 'pos' in data and isinstance(data['pos'], (list, tuple)):
            #         data['pos_str'] = str(data['pos']) # Convert to string
            #         del data['pos'] # Remove original complex type

            nx.write_gexf(self.manager.g, str(save_path), version="1.2draft")
            self.logger.info(f"Graph successfully saved to {save_path}")
        except ImportError:
            # This should ideally not happen if lxml is installed, but good practice
            self.logger.error(
                "Saving to GEXF requires the 'lxml' library. Please install it."
            )
            raise GraphIOError("Missing 'lxml' library required for GEXF export.")
        except Exception as e:
            self.logger.error(f"Failed to save graph to GEXF file '{save_path}': {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphIOError(f"Error writing GEXF file: {e}", e) from e

    @eidosian()
    def load_from_gexf(self, path: Optional[str] = None) -> None:
        """
        Load graph data from a GEXF file, replacing the current graph.

        Reads a graph from the specified GEXF file path. This operation
        replaces the existing graph in the GraphManager. It attempts to
        reconstruct the internal term-to-ID mapping based on loaded node data.

        Args:
            path: The file path to load the GEXF file from. If None, uses the
                  default path from the GraphConfig.

        Raises:
            GraphIOError: If the file doesn't exist, cannot be parsed, or if
                          loading fails.
            FileNotFoundError: If the specified GEXF file does not exist.
        """
        load_path_str = path or self._config.default_export_path
        load_path = Path(load_path_str)

        # Ensure the path includes the .gexf extension if not provided
        if load_path.suffix.lower() != ".gexf":
            load_path = load_path.with_suffix(".gexf")

        if not load_path.exists():
            self.logger.error(f"GEXF file not found at specified path: {load_path}")
            raise FileNotFoundError(f"GEXF file not found: {load_path}")
        if not load_path.is_file():
            self.logger.error(f"Specified path is not a file: {load_path}")
            raise GraphIOError(f"Path is not a file: {load_path}")

        self.logger.info(f"Loading graph from GEXF: {load_path}")

        try:
            # Load the graph using NetworkX
            loaded_graph = nx.read_gexf(str(load_path))

            # Replace the manager's graph
            self.manager.g = loaded_graph
            self.logger.info(
                f"Loaded graph with {loaded_graph.number_of_nodes()} nodes and {loaded_graph.number_of_edges()} edges."
            )

            # Rebuild internal mappings and potentially positions
            self.manager._term_to_id.clear()
            self.manager._positions.clear()
            self.manager._relationship_counts.clear()

            has_positions = False
            for node_id, data in loaded_graph.nodes(data=True):
                term = data.get("term")
                # GEXF loads node IDs as strings, ensure consistency if needed
                # current_node_id = int(node_id) # Assuming IDs were saved as ints
                current_node_id = (
                    node_id  # Keep as loaded type unless conversion needed
                )

                if term:
                    self.manager._term_to_id[str(term).lower()] = current_node_id
                else:
                    self.logger.warning(
                        f"Node '{current_node_id}' loaded from GEXF is missing 'term' attribute."
                    )

                # Attempt to load positions if present (GEXF might store viz attributes)
                pos_x = data.get("viz", {}).get("position", {}).get("x")
                pos_y = data.get("viz", {}).get("position", {}).get("y")
                pos_z = data.get("viz", {}).get("position", {}).get("z")  # For 3D

                if pos_x is not None and pos_y is not None:
                    if pos_z is not None:  # 3D position
                        self.manager._positions[current_node_id] = (
                            float(pos_x),
                            float(pos_y),
                            float(pos_z),
                        )
                        self.manager.dimensions = 3  # Assume 3D if z is present
                    else:  # 2D position
                        self.manager._positions[current_node_id] = (
                            float(pos_x),
                            float(pos_y),
                        )
                        # Only set to 2D if no 3D positions found yet
                        if not has_positions or self.manager.dimensions != 3:
                            self.manager.dimensions = 2
                    has_positions = True

            # Recalculate relationship counts
            for u, v, data in loaded_graph.edges(data=True):
                rel_type = data.get("relationship")
                if rel_type:
                    self.manager._relationship_counts[rel_type] = (
                        self.manager._relationship_counts.get(rel_type, 0) + 1
                    )

            # If positions weren't loaded from GEXF, compute layout
            if not has_positions and loaded_graph.number_of_nodes() > 0:
                self.logger.info(
                    "No position data found in GEXF, computing default layout."
                )
                self.manager.layout.compute_layout()
            elif has_positions:
                self.logger.info(
                    f"Loaded {len(self.manager._positions)} node positions from GEXF."
                )

        except ImportError:
            self.logger.error(
                "Loading from GEXF requires the 'lxml' library. Please install it."
            )
            raise GraphIOError("Missing 'lxml' library required for GEXF import.")
        except Exception as e:
            self.logger.error(f"Failed to load graph from GEXF file '{load_path}': {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            # Clear graph state on load failure to avoid inconsistent state
            self.manager.g.clear()
            self.manager._term_to_id.clear()
            self.manager._positions.clear()
            self.manager._relationship_counts.clear()
            raise GraphIOError(f"Error reading GEXF file: {e}", e) from e

    @eidosian()
    def export_subgraph(
        self, term: Term, depth: int = 1, output_path: Optional[str] = None
    ) -> str:
        """
        Extract and save a subgraph centered around a specific term.

        Performs a breadth-first search starting from the node corresponding
        to the given term, up to the specified depth. The resulting subgraph
        is then saved to a GEXF file.

        Args:
            term: The central term for the subgraph.
            depth: The maximum distance (number of hops) from the central term
                   to include in the subgraph. Defaults to 1.
            output_path: The file path to save the subgraph GEXF file. If None,
                         a default path is generated based on the term.

        Returns:
            str: The absolute path where the subgraph GEXF file was saved.

        Raises:
            NodeNotFoundError: If the specified term is not found in the graph.
            GraphIOError: If saving the subgraph fails.
            ValueError: If depth is negative.
        """
        if depth < 0:
            raise ValueError("Subgraph depth cannot be negative.")

        self.logger.info(f"Exporting subgraph for term '{term}' with depth {depth}.")

        # Find the starting node ID using the manager's query capability
        start_node_id = self.manager.query.get_node_id(term)
        if start_node_id is None:
            raise NodeNotFoundError(f"Term '{term}' not found in the graph.")

        # Extract the subgraph using NetworkX's ego_graph
        # ego_graph includes the center node and all nodes within the radius (depth)
        subgraph_nodes = nx.ego_graph(
            self.manager.g, start_node_id, radius=depth
        ).nodes()
        subgraph = self.manager.g.subgraph(
            subgraph_nodes
        ).copy()  # Create a copy to avoid modifying the original

        self.logger.info(
            f"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges."
        )

        if subgraph.number_of_nodes() == 0:
            self.logger.warning(
                f"Subgraph for '{term}' is empty (only the node itself?). Skipping export."
            )
            # Return an empty path or raise error based on desired behavior
            return ""

        # Determine the output path
        if output_path:
            save_path = Path(output_path)
        else:
            # Generate a default path in the configured export directory
            safe_term_name = "".join(c if c.isalnum() else "_" for c in term)
            filename = f"subgraph_{safe_term_name}_depth{depth}.gexf"
            save_path = self._config.get_export_path / filename

        # Ensure the target directory exists
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(
                f"Failed to create directory {save_path.parent} for subgraph export: {e}"
            )
            raise GraphIOError(
                f"Could not create directory for subgraph file: {save_path.parent}", e
            ) from e

        # Ensure .gexf extension
        if save_path.suffix.lower() != ".gexf":
            save_path = save_path.with_suffix(".gexf")

        self.logger.info(f"Saving subgraph to GEXF: {save_path}")

        try:
            # Save the extracted subgraph
            nx.write_gexf(subgraph, str(save_path), version="1.2draft")
            self.logger.info(f"Subgraph successfully saved to {save_path}")
            return str(save_path.resolve())  # Return absolute path
        except ImportError:
            self.logger.error("Saving to GEXF requires the 'lxml' library.")
            raise GraphIOError("Missing 'lxml' library required for GEXF export.")
        except Exception as e:
            self.logger.error(
                f"Failed to save subgraph to GEXF file '{save_path}': {e}"
            )
            self.logger.debug(f"Traceback: {traceback.format_exc()}", exc_info=True)
            raise GraphIOError(f"Error writing subgraph GEXF file: {e}", e) from e
