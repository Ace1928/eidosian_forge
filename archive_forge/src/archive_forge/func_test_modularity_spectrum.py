import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
def test_modularity_spectrum(self):
    """Modularity eigenvalues"""
    evals = np.array([-1.5, 0.0, 0.0])
    e = sorted(nx.modularity_spectrum(self.P))
    np.testing.assert_almost_equal(e, evals)
    evals = np.array([-0.5, 0.0, 0.0])
    e = sorted(nx.modularity_spectrum(self.DG))
    np.testing.assert_almost_equal(e, evals)