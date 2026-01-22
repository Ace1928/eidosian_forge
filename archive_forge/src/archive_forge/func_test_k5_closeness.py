import pytest
import networkx as nx
def test_k5_closeness(self):
    c = nx.closeness_centrality(self.K5)
    d = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    for n in sorted(self.K5):
        assert c[n] == pytest.approx(d[n], abs=0.001)