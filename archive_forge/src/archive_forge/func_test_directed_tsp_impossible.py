import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_directed_tsp_impossible():
    """
    Test the asadpour algorithm with a graph without a hamiltonian circuit
    """
    pytest.importorskip('numpy')
    edges = [(0, 1, 10), (0, 2, 11), (0, 3, 12), (1, 2, 4), (1, 3, 6), (2, 1, 3), (2, 3, 2), (3, 1, 5), (3, 2, 1)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    pytest.raises(nx.NetworkXError, nx_app.traveling_salesman_problem, G)