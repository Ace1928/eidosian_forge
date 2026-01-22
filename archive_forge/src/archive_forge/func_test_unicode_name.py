import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_unicode_name(self):
    """Tests that using a Unicode string can correctly indicate
        Borůvka's algorithm.
        """
    edges = nx.minimum_spanning_edges(self.G, algorithm='borůvka')
    actual = sorted(((min(u, v), max(u, v), d) for u, v, d in edges))
    assert edges_equal(actual, self.minimum_spanning_edgelist)