import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nan_weights(self):
    G = self.G
    G.add_edge(0, 12, weight=float('nan'))
    edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=True)
    actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
    expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
    assert edges_equal(actual, expected)
    edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=False)
    with pytest.raises(ValueError):
        list(edges)
    edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False)
    with pytest.raises(ValueError):
        list(edges)