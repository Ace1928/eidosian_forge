import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_directed_modularity(self):
    """Directed Modularity matrix"""
    B = np.array([[-0.2, 0.6, 0.8, -0.4, -0.4, -0.4], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.7, 0.4, -0.3, -0.6, 0.4, -0.6], [-0.2, -0.4, -0.2, -0.4, 0.6, 0.6], [-0.2, -0.4, -0.2, 0.6, -0.4, 0.6], [-0.1, -0.2, -0.1, 0.8, -0.2, -0.2]])
    node_permutation = [5, 1, 2, 3, 4, 6]
    idx_permutation = [4, 0, 1, 2, 3, 5]
    mm = nx.directed_modularity_matrix(self.DG, nodelist=sorted(self.DG))
    np.testing.assert_equal(mm, B)
    np.testing.assert_equal(nx.directed_modularity_matrix(self.DG, nodelist=node_permutation), B[np.ix_(idx_permutation, idx_permutation)])