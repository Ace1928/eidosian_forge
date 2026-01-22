import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_disconnected(self):
    G = nx.Graph([(0, 1, {'weight': 1}), (2, 3, {'weight': 2})])
    T = nx.minimum_spanning_tree(G, algorithm=self.algo)
    assert nodes_equal(list(T), list(range(4)))
    assert edges_equal(list(T.edges()), [(0, 1), (2, 3)])