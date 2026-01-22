import pytest
import networkx as nx
def test_krackhardt_load(self):
    G = self.K
    c = nx.load_centrality(G)
    d = {0: 0.023, 1: 0.023, 2: 0.0, 3: 0.102, 4: 0.0, 5: 0.231, 6: 0.231, 7: 0.389, 8: 0.222, 9: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)