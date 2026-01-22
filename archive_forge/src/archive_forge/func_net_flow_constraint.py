import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def net_flow_constraint(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> Hamiltonian:
    """Calculates the `net flow constraint <https://doi.org/10.1080/0020739X.2010.526248>`__
    Hamiltonian for the maximum-weighted cycle problem.

    Given a subset of edges in a directed graph, the net-flow constraint imposes that the number of
    edges leaving any given node is equal to the number of edges entering the node, i.e.,

    .. math:: \\sum_{j, (i, j) \\in E} x_{ij} = \\sum_{j, (j, i) \\in E} x_{ji},

    for all nodes :math:`i`, where :math:`E` are the edges of the graph and :math:`x_{ij}` is a
    binary number that selects whether to include the edge :math:`(i, j)`.

    A set of edges has zero net flow whenever the following Hamiltonian is minimized:

    .. math::

        \\sum_{i \\in V} \\left((d_{i}^{\\rm out} - d_{i}^{\\rm in})\\mathbb{I} -
        \\sum_{j, (i, j) \\in E} Z_{ij} + \\sum_{j, (j, i) \\in E} Z_{ji} \\right)^{2},

    where :math:`V` are the graph vertices, :math:`d_{i}^{\\rm out}` and :math:`d_{i}^{\\rm in}` are
    the outdegree and indegree, respectively, of node :math:`i` and :math:`Z_{ij}` is a qubit
    Pauli-Z matrix acting upon the wire specified by the pair :math:`(i, j)`. Mapping from edges to
    wires can be achieved using :func:`~.edges_to_wires`.


    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the net-flow constraint Hamiltonian

    Raises:
        ValueError: if the input graph is not directed
    """
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) and (not hasattr(graph, 'in_edges') or not hasattr(graph, 'out_edges')):
        raise ValueError('Input graph must be directed')
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    hamiltonian = Hamiltonian([], [])
    graph_nodes = graph.node_indexes() if isinstance(graph, rx.PyDiGraph) else graph.nodes
    for node in graph_nodes:
        hamiltonian += _inner_net_flow_constraint_hamiltonian(graph, node)
    return hamiltonian