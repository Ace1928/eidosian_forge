import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_normalized_laplacian_spectrum(self):
    """Normalized Laplacian eigenvalues"""
    evals = np.array([0, 0, 0.7712864461218, 1.5, 1.7287135538781])
    e = sorted(nx.normalized_laplacian_spectrum(self.G))
    np.testing.assert_almost_equal(e, evals)
    e = sorted(nx.normalized_laplacian_spectrum(self.WG, weight=None))
    np.testing.assert_almost_equal(e, evals)
    e = sorted(nx.normalized_laplacian_spectrum(self.WG))
    np.testing.assert_almost_equal(e, evals)
    e = sorted(nx.normalized_laplacian_spectrum(self.WG, weight='other'))
    np.testing.assert_almost_equal(e, evals)