import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('alg', (nx.pagerank, _pagerank_python, _pagerank_numpy))
def test_dangling_pagerank(self, alg):
    pr = alg(self.G, dangling=self.dangling_edges)
    for n in self.G:
        assert pr[n] == pytest.approx(self.G.dangling_pagerank[n], abs=0.0001)