import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_ascent_method_asymmetric():
    """
    Tests the ascent method using a truly asymmetric graph for which the
    solution has been brute forced
    """
    import networkx.algorithms.approximation.traveling_salesman as tsp
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G_array = np.array([[0, 26, 63, 59, 69, 31, 41], [62, 0, 91, 53, 75, 87, 47], [47, 82, 0, 90, 15, 9, 18], [68, 19, 5, 0, 58, 34, 93], [11, 58, 53, 55, 0, 61, 79], [88, 75, 13, 76, 98, 0, 40], [41, 61, 55, 88, 46, 45, 0]])
    solution_edges = [(0, 1), (1, 3), (3, 2), (2, 5), (5, 6), (4, 0), (6, 4)]
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    opt_hk, z_star = tsp.held_karp_ascent(G)
    assert round(opt_hk, 2) == 190.0
    solution = nx.DiGraph()
    solution.add_edges_from(solution_edges)
    assert nx.utils.edges_equal(z_star.edges, solution.edges)