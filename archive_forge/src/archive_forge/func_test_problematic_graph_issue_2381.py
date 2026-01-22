from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('method', methods)
def test_problematic_graph_issue_2381(self, method):
    pytest.importorskip('scipy')
    G = nx.path_graph(4)
    G.add_edges_from([(4, 2), (5, 1)])
    A = nx.laplacian_matrix(G)
    sigma = 0.438447187191
    ac = nx.algebraic_connectivity(G, tol=1e-12, method=method)
    assert ac == pytest.approx(sigma, abs=1e-07)
    x = nx.fiedler_vector(G, tol=1e-12, method=method)
    check_eigenvector(A, sigma, x)