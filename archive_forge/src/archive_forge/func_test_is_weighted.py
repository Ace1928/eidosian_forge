import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_is_weighted(self):
    G = nx.Graph()
    assert not nx.is_weighted(G)
    G = nx.path_graph(4)
    assert not nx.is_weighted(G)
    assert not nx.is_weighted(G, (2, 3))
    G.add_node(4)
    G.add_edge(3, 4, weight=4)
    assert not nx.is_weighted(G)
    assert nx.is_weighted(G, (3, 4))
    G = nx.DiGraph()
    G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
    assert nx.is_weighted(G)
    assert nx.is_weighted(G, ('1', '0'))
    G = G.to_undirected()
    assert nx.is_weighted(G)
    assert nx.is_weighted(G, ('1', '0'))
    pytest.raises(nx.NetworkXError, nx.is_weighted, G, (1, 2))