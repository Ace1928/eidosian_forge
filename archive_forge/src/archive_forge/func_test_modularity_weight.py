import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_modularity_weight(self):
    """Modularity matrix with weights"""
    B = np.array([[-1.125, 0.25, 0.25, 0.625, 0.0], [0.25, -0.5, 0.5, -0.25, 0.0], [0.25, 0.5, -0.5, -0.25, 0.0], [0.625, -0.25, -0.25, -0.125, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    G_weighted = self.G.copy()
    for n1, n2 in G_weighted.edges():
        G_weighted.edges[n1, n2]['weight'] = 0.5
    np.testing.assert_equal(nx.modularity_matrix(G_weighted), B)
    np.testing.assert_equal(nx.modularity_matrix(G_weighted, weight='weight'), 0.5 * B)