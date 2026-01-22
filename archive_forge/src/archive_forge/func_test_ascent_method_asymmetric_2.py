import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_ascent_method_asymmetric_2():
    """
    Tests the ascent method using a truly asymmetric graph for which the
    solution has been brute forced
    """
    import networkx.algorithms.approximation.traveling_salesman as tsp
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G_array = np.array([[0, 45, 39, 92, 29, 31], [72, 0, 4, 12, 21, 60], [81, 6, 0, 98, 70, 53], [49, 71, 59, 0, 98, 94], [74, 95, 24, 43, 0, 47], [56, 43, 3, 65, 22, 0]])
    solution_edges = [(0, 5), (5, 4), (1, 3), (3, 0), (2, 1), (4, 2)]
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    opt_hk, z_star = tsp.held_karp_ascent(G)
    assert round(opt_hk, 2) == 144.0
    solution = nx.DiGraph()
    solution.add_edges_from(solution_edges)
    assert nx.utils.edges_equal(z_star.edges, solution.edges)