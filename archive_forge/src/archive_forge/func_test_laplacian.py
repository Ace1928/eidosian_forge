import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
from networkx.generators.expanders import margulis_gabber_galil_graph
def test_laplacian(self):
    """Graph Laplacian"""
    NL = np.array([[3, -1, -1, -1, 0], [-1, 2, -1, 0, 0], [-1, -1, 2, 0, 0], [-1, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    WL = 0.5 * NL
    OL = 0.3 * NL
    np.testing.assert_equal(nx.laplacian_matrix(self.G).todense(), NL)
    np.testing.assert_equal(nx.laplacian_matrix(self.MG).todense(), NL)
    np.testing.assert_equal(nx.laplacian_matrix(self.G, nodelist=[0, 1]).todense(), np.array([[1, -1], [-1, 1]]))
    np.testing.assert_equal(nx.laplacian_matrix(self.WG).todense(), WL)
    np.testing.assert_equal(nx.laplacian_matrix(self.WG, weight=None).todense(), NL)
    np.testing.assert_equal(nx.laplacian_matrix(self.WG, weight='other').todense(), OL)