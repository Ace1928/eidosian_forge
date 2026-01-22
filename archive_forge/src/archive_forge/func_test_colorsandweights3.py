import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_colorsandweights3(self):
    self.g1.add_edge('A', 'B', weight=2)
    assert not nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)