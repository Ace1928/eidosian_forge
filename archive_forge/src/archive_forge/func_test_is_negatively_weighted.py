import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_is_negatively_weighted(self):
    G = nx.Graph()
    assert not nx.is_negatively_weighted(G)
    G.add_node(1)
    G.add_nodes_from([2, 3, 4, 5])
    assert not nx.is_negatively_weighted(G)
    G.add_edge(1, 2, weight=4)
    assert not nx.is_negatively_weighted(G, (1, 2))
    G.add_edges_from([(1, 3), (2, 4), (2, 6)])
    G[1][3]['color'] = 'blue'
    assert not nx.is_negatively_weighted(G)
    assert not nx.is_negatively_weighted(G, (1, 3))
    G[2][4]['weight'] = -2
    assert nx.is_negatively_weighted(G, (2, 4))
    assert nx.is_negatively_weighted(G)
    G = nx.DiGraph()
    G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -2), ('0', '2', 2), ('1', '2', -3), ('2', '3', 1)])
    assert nx.is_negatively_weighted(G)
    assert not nx.is_negatively_weighted(G, ('0', '3'))
    assert nx.is_negatively_weighted(G, ('1', '0'))
    pytest.raises(nx.NetworkXError, nx.is_negatively_weighted, G, (1, 4))