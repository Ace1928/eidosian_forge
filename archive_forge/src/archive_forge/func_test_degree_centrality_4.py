import pytest
import networkx as nx
def test_degree_centrality_4(self):
    d = nx.degree_centrality(self.F)
    names = sorted(self.F.nodes())
    dcs = [0.071, 0.214, 0.143, 0.214, 0.214, 0.071, 0.286, 0.071, 0.429, 0.071, 0.214, 0.214, 0.143, 0.286, 0.214]
    exact = dict(zip(names, dcs))
    for n, dc in d.items():
        assert exact[n] == pytest.approx(float(f'{dc:.3f}'), abs=1e-07)