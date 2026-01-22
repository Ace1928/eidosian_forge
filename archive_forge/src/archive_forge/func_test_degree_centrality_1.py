import pytest
import networkx as nx
def test_degree_centrality_1(self):
    d = nx.degree_centrality(self.K5)
    exact = dict(zip(range(5), [1] * 5))
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)