import math
import pytest
import networkx as nx
def test_katz_centrality_unweighted(self):
    H = self.H
    alpha = self.H.alpha
    p = nx.katz_centrality_numpy(H, alpha, weight='weight')
    for a, b in zip(list(p.values()), self.H.evc):
        assert a == pytest.approx(b, abs=1e-07)