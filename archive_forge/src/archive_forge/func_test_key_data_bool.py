import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_key_data_bool(self):
    """Tests that the keys and data values are included in
        MST edges based on whether keys and data parameters are
        true or false"""
    G = nx.MultiGraph()
    G.add_edge(1, 2, key=1, weight=2)
    G.add_edge(1, 2, key=2, weight=3)
    G.add_edge(3, 2, key=1, weight=2)
    G.add_edge(3, 1, key=1, weight=4)
    mst_edges = nx.minimum_spanning_edges(G, algorithm=self.algo, keys=True, data=False)
    assert edges_equal([(1, 2, 1), (2, 3, 1)], list(mst_edges))
    mst_edges = nx.minimum_spanning_edges(G, algorithm=self.algo, keys=False, data=True)
    assert edges_equal([(1, 2, {'weight': 2}), (2, 3, {'weight': 2})], list(mst_edges))
    mst_edges = nx.minimum_spanning_edges(G, algorithm=self.algo, keys=False, data=False)
    assert edges_equal([(1, 2), (2, 3)], list(mst_edges))
    mst_edges = nx.minimum_spanning_edges(G, algorithm=self.algo, keys=True, data=True)
    assert edges_equal([(1, 2, 1, {'weight': 2}), (2, 3, 1, {'weight': 2})], list(mst_edges))