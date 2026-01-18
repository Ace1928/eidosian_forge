"""
Manages graph layout computation and node positioning.

Encapsulates logic for applying various layout algorithms (2D and 3D)
to the knowledge graph, handling incremental updates, and storing
position data. Adheres to Eidosian principles of modularity, precision,
and adaptability.

Architecture:
    ┌──────────────────┐      ┌──────────────────┐
    │  GraphManager    │◄────►│   GraphLayout    │
    │ (Orchestrator)   │      │ (Position Calc & │
    └────────┬─────────┘      │  Algorithm Mgmt) │
             │                └──────────────────┘
             ▼
    ┌──────────────────┐
    │    NetworkX      │
    │ (Layout Algos)   │
    └──────────────────┘
"""

from __future__ import annotations

import functools  # Import functools
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, cast

import networkx as nx

# Import necessary components
from word_forge.exceptions import GraphLayoutError
from word_forge.graph.graph_config import (
    LayoutAlgorithm,
    Position,
    PositionDict,
    WordId,  # Alias to avoid naming conflict
)

# Type hint for the main GraphManager to avoid circular imports
if TYPE_CHECKING:
    from .graph_manager import GraphManager


class GraphLayout:
    """
    Computes and manages node positions for graph visualization.

    Supports various NetworkX layout algorithms in both 2D and 3D,
    adapting based on the GraphManager's dimension setting. Provides methods
    for full layout computation and incremental updates for newly added nodes.

    Attributes:
        manager: Reference to the main GraphManager for state access.
        logger: Logger instance for this module.
    """

    def __init__(self, manager: GraphManager) -> None:
        """
        Initialize the GraphLayout with a reference to the GraphManager.

        Args:
            manager: The orchestrating GraphManager instance.
        """
        self.manager: GraphManager = manager
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Use config from manager for consistency
        self._config = self.manager.config

    def compute_layout(self, algorithm: Optional[LayoutAlgorithm] = None) -> None:
        """
        Compute node positions for the entire graph using a specified algorithm.

        Applies the chosen layout algorithm (or the default from config) to
        all nodes in the manager's graph. Handles both 2D and 3D layouts based
        on the manager's dimension setting. Stores the computed positions in
        the manager's `_positions` dictionary.

        Args:
            algorithm: The layout algorithm to use (e.g., 'force_directed').
                       If None, uses the default from GraphConfig.

        Raises:
            GraphLayoutError: If the layout computation fails or the graph is empty.
            AttributeError: If an invalid layout algorithm is specified.
        """
        if self.manager.g.number_of_nodes() == 0:
            self.logger.warning("Cannot compute layout for an empty graph.")
            self.manager._positions.clear()  # Ensure positions are cleared
            return

        layout_algo_name = algorithm or self._config.default_layout
        dimensions = self.manager.dimensions  # Get current dimensionality

        self.logger.info(
            f"Computing {dimensions}D graph layout using '{layout_algo_name}' algorithm."
        )

        try:
            # Ensure layout_algo_name is the string value if it's an Enum
            algo_str = (
                layout_algo_name.value
                if hasattr(layout_algo_name, "value")
                else layout_algo_name
            )
            layout_func = self._get_layout_function(
                algo_str, dimensions
            )  # Pass string value
            # Pass the graph instance from the manager
            computed_positions: Dict[WordId, Position] = layout_func(self.manager.g)

            # Store the computed positions in the manager's state
            self.manager._positions = cast(PositionDict, computed_positions)
            self.logger.info(
                f"Layout computation complete. Stored {len(self.manager._positions)} node positions."
            )

        except AttributeError as e:
            # Use algo_str in error message
            self.logger.error(f"Invalid layout algorithm specified: '{algo_str}'. {e}")
            raise GraphLayoutError(
                f"Layout algorithm '{algo_str}' not found or failed.", e
            ) from e
        except Exception as e:
            # Use algo_str in error message
            self.logger.error(
                f"Error during layout computation using '{algo_str}': {e}",
                exc_info=True,  # Add traceback for unexpected errors
            )
            # Optionally clear positions on error, or leave them potentially partially computed
            # self.manager._positions.clear()
            raise GraphLayoutError(f"Layout computation failed: {e}", e) from e

    def update_layout_incrementally(self, new_node_ids: List[WordId]) -> None:
        """
        Update layout incrementally, focusing on positioning new nodes.

        Currently, this often involves recomputing the layout for stability,
        especially with force-directed algorithms. Future enhancements could
        implement true incremental updates for specific algorithms if feasible
        and beneficial. For now, it primarily recalculates the full layout.

        Args:
            new_node_ids: A list of IDs for the newly added nodes.
                          (Currently unused, but kept for future incremental logic).

        Raises:
            GraphLayoutError: If the layout computation fails.
        """
        if not new_node_ids:
            self.logger.debug("No new nodes provided for incremental layout update.")
            return

        self.logger.info(
            f"Received {len(new_node_ids)} new nodes. Updating layout incrementally."
        )

        default_algo = self._config.default_layout
        algo_str = (
            default_algo.value if hasattr(default_algo, "value") else default_algo
        )

        if algo_str != "force_directed" or not self.manager._positions:
            # Fallback to full recompute for unsupported algorithms or missing positions
            self.logger.debug(
                "Incremental update not supported for this layout. Recomputing full layout."
            )
            self.compute_layout(algorithm=algo_str)
            return

        try:
            fixed_nodes = [n for n in self.manager.g.nodes() if n not in new_node_ids]
            pos_init = {
                n: self.manager._positions[n]
                for n in fixed_nodes
                if n in self.manager._positions
            }
            k_value = getattr(self._config, "layout_k", None)
            iterations = getattr(self._config, "layout_iterations", 50)
            updated_pos = nx.spring_layout(
                self.manager.g,
                pos=pos_init,
                fixed=fixed_nodes,
                dim=self.manager.dimensions,
                k=k_value,
                iterations=iterations,
            )
            self.manager._positions.update(cast(PositionDict, updated_pos))
            self.logger.info(
                f"Incremental layout update complete. Total nodes positioned: {len(self.manager._positions)}."
            )
        except Exception as e:
            self.logger.error(
                f"Incremental layout update failed: {e}",
                exc_info=True,
            )
            raise GraphLayoutError(f"Incremental layout update failed: {e}") from e

    def _get_layout_function(self, algorithm_name: str, dimensions: int) -> callable:
        """
        Retrieve the appropriate NetworkX layout function based on name and dimension.
        Ensures correct parameters are passed to the underlying layout function.

        Args:
            algorithm_name: The name of the layout algorithm.
            dimensions: The desired number of dimensions (2 or 3).

        Returns:
            callable: A function that takes a graph `G` and returns positions.

        Raises:
            AttributeError: If the algorithm name is invalid.
            ValueError: If the dimensions are not 2 or 3.
        """
        if dimensions not in [2, 3]:
            raise ValueError(
                f"Invalid number of dimensions specified: {dimensions}. Must be 2 or 3."
            )

        # Common parameters for spring_layout
        k_value = getattr(self._config, "layout_k", None)
        iterations = getattr(self._config, "layout_iterations", 50)

        # Define base layout functions
        layout_map_base: Dict[str, callable] = {
            "force_directed": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "circular": nx.circular_layout,
            "hierarchical": lambda G: nx.nx_agraph.graphviz_layout(G, prog="dot"),
            "radial": lambda G: nx.nx_agraph.graphviz_layout(G, prog="twopi"),
            "grid": nx.spring_layout,  # Fallback for grid
        }

        try:
            base_func = layout_map_base[algorithm_name]

            # Handle dimension-specific logic and parameters
            if algorithm_name == "force_directed":
                # Use functools.partial to pre-set arguments for spring_layout
                return functools.partial(
                    base_func, dim=dimensions, k=k_value, iterations=iterations
                )
            elif algorithm_name == "spectral":
                if dimensions == 3:
                    try:
                        # Attempt 3D spectral layout
                        return functools.partial(base_func, dim=3)
                    except TypeError:
                        self.logger.warning(
                            "Current NetworkX spectral_layout doesn't support dim=3. Falling back to 3D spring layout."
                        )
                        # Fallback to 3D spring layout
                        return functools.partial(
                            nx.spring_layout, dim=3, k=k_value, iterations=iterations
                        )
                else:  # dimensions == 2
                    return base_func  # spectral_layout defaults to 2D
            elif algorithm_name == "circular":
                # Circular layout is inherently 2D. For 3D, use spring as fallback.
                if dimensions == 3:
                    self.logger.debug(
                        "Circular layout requested for 3D, using 3D spring layout as fallback."
                    )
                    return functools.partial(
                        nx.spring_layout, dim=3, k=k_value, iterations=iterations
                    )
                else:
                    return base_func
            elif algorithm_name in ["hierarchical", "radial"]:
                if dimensions == 3:
                    self.logger.warning(
                        f"Layout '{algorithm_name}' is 2D only. Falling back to 3D spring layout."
                    )
                    return functools.partial(
                        nx.spring_layout, dim=3, k=k_value, iterations=iterations
                    )
                else:
                    # Check for pygraphviz dependency
                    try:
                        import pygraphviz  # noqa: F401

                        return base_func  # Return the lambda defined in layout_map_base
                    except ImportError:
                        self.logger.warning(
                            f"Layout '{algorithm_name}' requires pygraphviz. Falling back to 'force_directed'."
                        )
                        self.logger.warning("Install with: pip install pygraphviz")
                        # Fallback to 2D spring layout
                        return functools.partial(
                            nx.spring_layout, dim=2, k=k_value, iterations=iterations
                        )
            elif algorithm_name == "grid":
                self.logger.debug(
                    "Grid layout requested, using spring layout as fallback."
                )
                return functools.partial(
                    nx.spring_layout, dim=dimensions, k=k_value, iterations=iterations
                )
            else:
                # Should not be reached if algorithm_name is in layout_map_base
                raise AttributeError(f"Unhandled layout algorithm: {algorithm_name}")

        except KeyError:
            self.logger.error(f"Layout algorithm '{algorithm_name}' is not supported.")
            raise AttributeError(f"Unsupported layout algorithm: {algorithm_name}")
        except ImportError as ie:
            # Catch potential import errors from nx_agraph
            self.logger.error(
                f"Layout algorithm '{algorithm_name}' failed due to missing dependency: {ie}. Falling back to force_directed."
            )
            # Fallback to appropriate dimension spring layout
            return functools.partial(
                nx.spring_layout, dim=dimensions, k=k_value, iterations=iterations
            )

    def _apply_layout(self) -> None:
        """
        Applies the computed positions to the graph nodes.

        Deprecated/Internal: Positions are now stored directly in
        `self.manager._positions`. This method might be repurposed if
        node attributes need direct updating in the future.
        """
        # This method is less relevant now as positions are stored centrally
        # in self.manager._positions. Keeping as a placeholder or for future use
        # if direct node attribute updates become necessary.
        self.logger.debug(
            "Layout positions are stored centrally; direct application to node attributes skipped."
        )
        # Example of direct application if needed later:
        # if self.manager._positions:
        #     for node_id, pos in self.manager._positions.items():
        #         if node_id in self.manager.g:
        #             self.manager.g.nodes[node_id['pos'] = pos
