import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_add_star(self):
    G = self.G.copy()
    nlist = [12, 13, 14, 15]
    nx.add_star(G, nlist)
    assert edges_equal(G.edges(nlist), [(12, 13), (12, 14), (12, 15)])
    G = self.G.copy()
    nx.add_star(G, nlist, weight=2.0)
    assert edges_equal(G.edges(nlist, data=True), [(12, 13, {'weight': 2.0}), (12, 14, {'weight': 2.0}), (12, 15, {'weight': 2.0})])
    G = self.G.copy()
    nlist = [12]
    nx.add_star(G, nlist)
    assert nodes_equal(G, list(self.G) + nlist)
    G = self.G.copy()
    nlist = []
    nx.add_star(G, nlist)
    assert nodes_equal(G.nodes, self.Gnodes)
    assert edges_equal(G.edges, self.G.edges)