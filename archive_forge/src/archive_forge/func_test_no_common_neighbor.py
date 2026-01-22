import math
from functools import partial
import pytest
import networkx as nx
def test_no_common_neighbor(self):
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 0
    self.test(G, [(0, 1)], [(0, 1, 0)])