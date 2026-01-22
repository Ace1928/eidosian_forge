import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_asadpour_tsp():
    """
    Test the complete asadpour tsp algorithm with the fractional, symmetric
    Held Karp solution. This test also uses an incomplete graph as input.
    """
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    edge_list = [(0, 1, 100), (0, 2, 100), (0, 5, 1), (1, 2, 100), (1, 4, 1), (2, 3, 1), (3, 4, 100), (3, 5, 100), (4, 5, 100), (1, 0, 100), (2, 0, 100), (5, 0, 1), (2, 1, 100), (4, 1, 1), (3, 2, 1), (4, 3, 100), (5, 3, 100), (5, 4, 100)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edge_list)

    def fixed_asadpour(G, weight):
        return nx_app.asadpour_atsp(G, weight, 19)
    tour = nx_app.traveling_salesman_problem(G, weight='weight', method=fixed_asadpour)
    expected_tours = [[1, 4, 5, 0, 2, 3, 2, 1], [3, 2, 0, 1, 4, 5, 3]]
    assert tour in expected_tours