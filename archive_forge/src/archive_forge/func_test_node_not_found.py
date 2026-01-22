import math
from functools import partial
import pytest
import networkx as nx
def test_node_not_found(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])
    G.nodes[0]['community'] = 0
    G.nodes[1]['community'] = 1
    G.nodes[2]['community'] = 0
    G.nodes[3]['community'] = 0
    assert pytest.raises(nx.NodeNotFound, self.func, G, [(0, 4)])