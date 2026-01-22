import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_selfloop_digraph(self):
    G = nx.DiGraph([(1, 1)])
    M = nx.to_scipy_sparse_array(G)
    np.testing.assert_equal(M.toarray(), np.array([[1]]))
    G.add_edges_from([(2, 3), (3, 4)])
    M = nx.to_scipy_sparse_array(G, nodelist=[2, 3, 4])
    np.testing.assert_equal(M.toarray(), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))