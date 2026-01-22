import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def draw_gridlike(graph: nx.Graph, ax: Optional[plt.Axes]=None, tilted: bool=True, **kwargs) -> Dict[Any, Tuple[int, int]]:
    """Draw a grid-like graph using Matplotlib.

    This wraps nx.draw_networkx to produce a matplotlib drawing of the graph. Nodes
    should be two-dimensional gridlike objects.

    Args:
        graph: A NetworkX graph whose nodes are (row, column) coordinates or cirq.GridQubits.
        ax: Optional matplotlib axis to use for drawing.
        tilted: If True, directly position as (row, column); otherwise,
            rotate 45 degrees to accommodate google-style diagonal grids.
        **kwargs: Additional arguments to pass to `nx.draw_networkx`.

    Returns:
        A positions dictionary mapping nodes to (x, y) coordinates suitable for future calls
        to NetworkX plotting functionality.
    """
    if ax is None:
        ax = plt.gca()
    if tilted:
        pos = {node: (y, -x) for node, (x, y) in _node_and_coordinates(graph.nodes)}
    else:
        pos = {node: (x + y, y - x) for node, (x, y) in _node_and_coordinates(graph.nodes)}
    nx.draw_networkx(graph, pos=pos, ax=ax, **kwargs)
    ax.axis('equal')
    return pos