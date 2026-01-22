import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_colorsandweights1(self):
    iso = nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)
    assert not iso