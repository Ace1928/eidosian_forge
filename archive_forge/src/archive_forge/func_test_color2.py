import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_color2(self):
    self.g1.nodes['A']['color'] = 'blue'
    assert nx.is_isomorphic(self.g1, self.g2, node_match=self.nm)