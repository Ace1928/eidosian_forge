import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_freeze(self):
    G = nx.freeze(self.G)
    assert G.frozen
    pytest.raises(nx.NetworkXError, G.add_node, 1)
    pytest.raises(nx.NetworkXError, G.add_nodes_from, [1])
    pytest.raises(nx.NetworkXError, G.remove_node, 1)
    pytest.raises(nx.NetworkXError, G.remove_nodes_from, [1])
    pytest.raises(nx.NetworkXError, G.add_edge, 1, 2)
    pytest.raises(nx.NetworkXError, G.add_edges_from, [(1, 2)])
    pytest.raises(nx.NetworkXError, G.remove_edge, 1, 2)
    pytest.raises(nx.NetworkXError, G.remove_edges_from, [(1, 2)])
    pytest.raises(nx.NetworkXError, G.clear_edges)
    pytest.raises(nx.NetworkXError, G.clear)