import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_isolated_node(self):
    edges = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7), (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)]
    G = nx.Graph()
    G.add_weighted_edges_from([(u + 1, v + 1, wt) for u, v, wt in edges])
    G.add_node(0)
    edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=True)
    actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
    shift = [(u + 1, v + 1) for u, v, d in self.minimum_spanning_edgelist]
    assert edges_equal(actual, shift)