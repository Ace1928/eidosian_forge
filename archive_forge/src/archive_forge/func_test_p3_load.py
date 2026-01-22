import pytest
import networkx as nx
def test_p3_load(self):
    G = self.P3
    c = nx.load_centrality(G)
    d = {0: 0.0, 1: 1.0, 2: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)
    c = nx.load_centrality(G, v=1)
    assert c == pytest.approx(1.0, abs=1e-07)
    c = nx.load_centrality(G, v=1, normalized=True)
    assert c == pytest.approx(1.0, abs=1e-07)