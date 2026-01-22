import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_cycle_C5(self):
    c = harmonic_centrality(self.C5)
    d = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 4}
    for n in sorted(self.C5):
        assert c[n] == pytest.approx(d[n], abs=0.001)