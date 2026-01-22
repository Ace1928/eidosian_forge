import math
from functools import partial
import pytest
import networkx as nx
def test_node_u_not_found(self):
    G = nx.Graph()
    G.add_edges_from([(1, 3), (2, 3)])
    assert pytest.raises(nx.NodeNotFound, self.func, G, [(0, 1)])