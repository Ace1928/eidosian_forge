import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def cycle_mixer(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> Hamiltonian:
    """Calculates the cycle-mixer Hamiltonian.

    Following methods outlined `here <https://arxiv.org/abs/1709.03489>`__, the
    cycle-mixer Hamiltonian preserves the set of valid cycles:

    .. math::
        \\frac{1}{4}\\sum_{(i, j)\\in E}
        \\left(\\sum_{k \\in V, k\\neq i, k\\neq j, (i, k) \\in E, (k, j) \\in E}
        \\left[X_{ij}X_{ik}X_{kj} +Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\\right]
        \\right)

    where :math:`E` are the edges of the directed graph. A valid cycle is defined as a subset of
    edges in :math:`E` such that all of the graph's nodes :math:`V` have zero net flow (see the
    :func:`~.net_flow_constraint` function).

    **Example**

    >>> import networkx as nx
    >>> g = nx.complete_graph(3).to_directed()
    >>> h_m = cycle_mixer(g)
    >>> print(h_m)
      (-0.25) [X0 Y1 Y5]
    + (-0.25) [X1 Y0 Y3]
    + (-0.25) [X2 Y3 Y4]
    + (-0.25) [X3 Y2 Y1]
    + (-0.25) [X4 Y5 Y2]
    + (-0.25) [X5 Y4 Y0]
    + (0.25) [X0 X1 X5]
    + (0.25) [Y0 Y1 X5]
    + (0.25) [Y0 X1 Y5]
    + (0.25) [X1 X0 X3]
    + (0.25) [Y1 Y0 X3]
    + (0.25) [Y1 X0 Y3]
    + (0.25) [X2 X3 X4]
    + (0.25) [Y2 Y3 X4]
    + (0.25) [Y2 X3 Y4]
    + (0.25) [X3 X2 X1]
    + (0.25) [Y3 Y2 X1]
    + (0.25) [Y3 X2 Y1]
    + (0.25) [X4 X5 X2]
    + (0.25) [Y4 Y5 X2]
    + (0.25) [Y4 X5 Y2]
    + (0.25) [X5 X4 X0]
    + (0.25) [Y5 Y4 X0]
    + (0.25) [Y5 X4 Y0]

    >>> import rustworkx as rx
    >>> g = rx.generators.directed_mesh_graph(3, [0,1,2])
    >>> h_m = cycle_mixer(g)
    >>> print(h_m)
      (-0.25) [X0 Y1 Y5]
    + (-0.25) [X1 Y0 Y3]
    + (-0.25) [X2 Y3 Y4]
    + (-0.25) [X3 Y2 Y1]
    + (-0.25) [X4 Y5 Y2]
    + (-0.25) [X5 Y4 Y0]
    + (0.25) [X0 X1 X5]
    + (0.25) [Y0 Y1 X5]
    + (0.25) [Y0 X1 Y5]
    + (0.25) [X1 X0 X3]
    + (0.25) [Y1 Y0 X3]
    + (0.25) [Y1 X0 Y3]
    + (0.25) [X2 X3 X4]
    + (0.25) [Y2 Y3 X4]
    + (0.25) [Y2 X3 Y4]
    + (0.25) [X3 X2 X1]
    + (0.25) [Y3 Y2 X1]
    + (0.25) [Y3 X2 Y1]
    + (0.25) [X4 X5 X2]
    + (0.25) [Y4 Y5 X2]
    + (0.25) [Y4 X5 Y2]
    + (0.25) [X5 X4 X0]
    + (0.25) [Y5 Y4 X0]
    + (0.25) [Y5 X4 Y0]

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the cycle-mixer Hamiltonian
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    hamiltonian = Hamiltonian([], [])
    graph_edges = sorted(graph.edge_list()) if isinstance(graph, rx.PyDiGraph) else graph.edges
    for edge in graph_edges:
        hamiltonian += _partial_cycle_mixer(graph, edge)
    return hamiltonian