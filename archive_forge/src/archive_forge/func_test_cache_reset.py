import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_cache_reset(self):
    G = self.K3.copy()
    old_adj = G.adj
    assert id(G.adj) == id(old_adj)
    G._adj = {}
    assert id(G.adj) != id(old_adj)
    old_nodes = G.nodes
    assert id(G.nodes) == id(old_nodes)
    G._node = {}
    assert id(G.nodes) != id(old_nodes)