import itertools
import functools
from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane.wires import Wires
def xy_mixer(graph: Union[nx.Graph, rx.PyGraph]):
    """Creates a generalized SWAP/XY mixer Hamiltonian.

    This mixer Hamiltonian is defined as:

    .. math:: H_M \\ = \\ \\frac{1}{2} \\displaystyle\\sum_{(i, j) \\in E(G)} X_i X_j \\ + \\ Y_i Y_j,

    for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
    wire respectively.

    This mixer was introduced in *From the Quantum Approximate Optimization Algorithm
    to a Quantum Alternating Operator Ansatz* by Stuart Hadfield, Zhihui Wang, Bryan O'Gorman,
    Eleanor G. Rieffel, Davide Venturelli, and Rupak Biswas `Algorithms 12.2 (2019) <https://doi.org/10.3390/a12020034>`__.

    Args:
        graph (nx.Graph or rx.PyGraph): A graph defining the collections of wires on which the Hamiltonian acts.

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> from networkx import Graph
    >>> graph = Graph([(0, 1), (1, 2)])
    >>> mixer_h = qaoa.xy_mixer(graph)
    >>> print(mixer_h)
      (0.5) [X0 X1]
    + (0.5) [Y0 Y1]
    + (0.5) [X1 X2]
    + (0.5) [Y1 Y2]

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1, ""), (1, 2, "")])
    >>> mixer_h = xy_mixer(graph)
    >>> print(mixer_h)
      (0.5) [X0 X1]
    + (0.5) [Y0 Y1]
    + (0.5) [X1 X2]
    + (0.5) [Y1 Y2]
    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}')
    is_rx = isinstance(graph, rx.PyGraph)
    edges = graph.edge_list() if is_rx else graph.edges
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i
    coeffs = 2 * [0.5 for e in edges]
    obs = []
    for node1, node2 in edges:
        obs.append(qml.X(get_nvalue(node1)) @ qml.X(get_nvalue(node2)))
        obs.append(qml.Y(get_nvalue(node1)) @ qml.Y(get_nvalue(node2)))
    return qml.Hamiltonian(coeffs, obs)