import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_edge_attr_dict(self):
    """Tests that the edge attribute dictionary of the two graphs is
        the same object.

        """
    for u, v in self.H.edges():
        assert self.G.edges[u, v] == self.H.edges[u, v]
    self.G.edges[0, 1]['name'] = 'foo'
    assert self.G.edges[0, 1]['name'] == self.H.edges[0, 1]['name']
    self.H.edges[3, 4]['name'] = 'bar'
    assert self.G.edges[3, 4]['name'] == self.H.edges[3, 4]['name']
    self.G.edges[0, 1]['name'] = 'edge01'
    self.H.edges[3, 4]['name'] = 'edge34'