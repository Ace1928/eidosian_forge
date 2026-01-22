import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('wrapper', [lambda x: x, dispatch_interface.convert])
def test_google_matrix(self, wrapper):
    G = wrapper(self.G)
    M = nx.google_matrix(G, alpha=0.9, nodelist=sorted(G))
    _, ev = np.linalg.eig(M.T)
    p = ev[:, 0] / ev[:, 0].sum()
    for a, b in zip(p, self.G.pagerank.values()):
        assert a == pytest.approx(b, abs=1e-07)