import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_prim_mst_edges_simple_graph(self):
    H = nx.Graph()
    H.add_edge(1, 2, key=2, weight=3)
    H.add_edge(3, 2, key=1, weight=2)
    H.add_edge(3, 1, key=1, weight=4)
    mst_edges = nx.minimum_spanning_edges(H, algorithm=self.algo, ignore_nan=True)
    assert edges_equal([(1, 2, {'key': 2, 'weight': 3}), (2, 3, {'key': 1, 'weight': 2})], list(mst_edges))