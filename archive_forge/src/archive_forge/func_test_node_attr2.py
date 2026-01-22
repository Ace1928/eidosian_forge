import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_node_attr2(self):
    G = self.K3.copy()
    a = {'foo': 'bar'}
    G.add_node(3, **a)
    assert nodes_equal(G.nodes(), [0, 1, 2, 3])
    assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {}), (2, {}), (3, {'foo': 'bar'})])