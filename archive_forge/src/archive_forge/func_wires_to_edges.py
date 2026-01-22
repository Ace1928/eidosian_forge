import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def wires_to_edges(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph]) -> Dict[int, Tuple]:
    """Maps the wires of a register of qubits to corresponding edges.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> wires_to_edges(g)
    {0: (0, 1),
     1: (0, 2),
     2: (0, 3),
     3: (1, 0),
     4: (1, 2),
     5: (1, 3),
     6: (2, 0),
     7: (2, 1),
     8: (2, 3),
     9: (3, 0),
     10: (3, 1),
     11: (3, 2)}

    >>> g = rx.generators.directed_mesh_graph(4, [0,1,2,3])
    >>> wires_to_edges(g)
    {0: (0, 1),
     1: (0, 2),
     2: (0, 3),
     3: (1, 0),
     4: (1, 2),
     5: (1, 3),
     6: (2, 0),
     7: (2, 1),
     8: (2, 3),
     9: (3, 0),
     10: (3, 1),
     11: (3, 2)}

    Args:
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the graph specifying possible edges

    Returns:
        Dict[Tuple, int]: a mapping from wires to graph edges
    """
    if isinstance(graph, nx.Graph):
        return {i: edge for i, edge in enumerate(graph.edges)}
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        gnodes = graph.nodes()
        return {i: (gnodes.index(e[0]), gnodes.index(e[1])) for i, e in enumerate(sorted(graph.edge_list()))}
    raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}')