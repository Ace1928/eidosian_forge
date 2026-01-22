import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def edges_to_wires(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph]) -> Dict[Tuple, int]:
    """Maps the edges of a graph to corresponding wires.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> edges_to_wires(g)
    {(0, 1): 0,
     (0, 2): 1,
     (0, 3): 2,
     (1, 0): 3,
     (1, 2): 4,
     (1, 3): 5,
     (2, 0): 6,
     (2, 1): 7,
     (2, 3): 8,
     (3, 0): 9,
     (3, 1): 10,
     (3, 2): 11}

    >>> g = rx.generators.directed_mesh_graph(4, [0,1,2,3])
    >>> edges_to_wires(g)
    {(0, 1): 0,
     (0, 2): 1,
     (0, 3): 2,
     (1, 0): 3,
     (1, 2): 4,
     (1, 3): 5,
     (2, 0): 6,
     (2, 1): 7,
     (2, 3): 8,
     (3, 0): 9,
     (3, 1): 10,
     (3, 2): 11}

    Args:
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the graph specifying possible edges

    Returns:
        Dict[Tuple, int]: a mapping from graph edges to wires
    """
    if isinstance(graph, nx.Graph):
        return {edge: i for i, edge in enumerate(graph.edges)}
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        gnodes = graph.nodes()
        return {(gnodes.index(e[0]), gnodes.index(e[1])): i for i, e in enumerate(sorted(graph.edge_list()))}
    raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}')