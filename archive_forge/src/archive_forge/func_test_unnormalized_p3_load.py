import pytest
import networkx as nx
def test_unnormalized_p3_load(self):
    G = self.P3
    c = nx.load_centrality(G, normalized=False)
    d = {0: 0.0, 1: 2.0, 2: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)