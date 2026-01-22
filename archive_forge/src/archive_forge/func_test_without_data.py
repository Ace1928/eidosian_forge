import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_without_data(self):
    edges = nx.minimum_spanning_edges(self.G, algorithm=self.algo, data=False)
    actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
    expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
    assert edges_equal(actual, expected)