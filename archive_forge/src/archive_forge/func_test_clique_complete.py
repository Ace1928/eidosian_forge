import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_clique_complete(self):
    c = harmonic_centrality(self.K5)
    d = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}
    for n in sorted(self.P3):
        assert c[n] == pytest.approx(d[n], abs=0.001)