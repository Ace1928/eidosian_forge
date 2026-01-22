import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edge_attr3(self):
    G = self.Graph()
    G.add_edges_from([(1, 2, {'weight': 32}), (3, 4, {'weight': 64})], foo='foo')
    assert edges_equal(G.edges(data=True), [(1, 2, {'foo': 'foo', 'weight': 32}), (3, 4, {'foo': 'foo', 'weight': 64})])
    G.remove_edges_from([(1, 2), (3, 4)])
    G.add_edge(1, 2, data=7, spam='bar', bar='foo')
    assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])