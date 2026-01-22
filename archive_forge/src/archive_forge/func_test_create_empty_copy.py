import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_create_empty_copy(self):
    G = nx.create_empty_copy(self.G, with_data=False)
    assert nodes_equal(G, list(self.G))
    assert G.graph == {}
    assert G._node == {}.fromkeys(self.G.nodes(), {})
    assert G._adj == {}.fromkeys(self.G.nodes(), {})
    G = nx.create_empty_copy(self.G)
    assert nodes_equal(G, list(self.G))
    assert G.graph == self.G.graph
    assert G._node == self.G._node
    assert G._adj == {}.fromkeys(self.G.nodes(), {})