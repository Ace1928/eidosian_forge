import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize(('values', 'name'), (({(0, 1): 1.0, (0, 2): 2.0}, 'weight'), ({(0, 1): {'weight': 1.0}, (0, 2): {'weight': 2.0}}, None)))
def test_set_edge_attributes_ignores_extra_edges(values, name):
    """If `values` is a dict or dict-of-dicts containing edges that are not in
    G, data associate with these edges should be ignored.
    """
    G = nx.Graph([(0, 1)])
    nx.set_edge_attributes(G, values, name)
    assert G[0][1]['weight'] == 1.0
    assert (0, 2) not in G.edges