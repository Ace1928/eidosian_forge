import math
from functools import partial
import pytest
import networkx as nx
def test_custom_community_attribute_name(self):
    G = nx.complete_graph(4)
    G.nodes[0]['cmty'] = 0
    G.nodes[1]['cmty'] = 0
    G.nodes[2]['cmty'] = 0
    G.nodes[3]['cmty'] = 0
    self.test(G, [(0, 3)], [(0, 3, 2 / self.delta)], community='cmty')