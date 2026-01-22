import pytest
import networkx as nx
def test_degree_centrality_2(self):
    d = nx.degree_centrality(self.P3)
    exact = {0: 0.5, 1: 1, 2: 0.5}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)