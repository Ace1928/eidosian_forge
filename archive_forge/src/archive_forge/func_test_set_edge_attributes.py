import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.DiGraph))
def test_set_edge_attributes(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    attr = 'hello'
    vals = 3
    nx.set_edge_attributes(G, vals, attr)
    assert G[0][1][attr] == vals
    assert G[1][2][attr] == vals
    G = nx.path_graph(3, create_using=graph_type)
    attr = 'hi'
    edges = [(0, 1), (1, 2)]
    vals = dict(zip(edges, range(len(edges))))
    nx.set_edge_attributes(G, vals, attr)
    assert G[0][1][attr] == 0
    assert G[1][2][attr] == 1
    G = nx.path_graph(3, create_using=graph_type)
    d = {'hi': 0, 'hello': 200}
    edges = [(0, 1)]
    vals = dict.fromkeys(edges, d)
    nx.set_edge_attributes(G, vals)
    assert G[0][1]['hi'] == 0
    assert G[0][1]['hello'] == 200
    assert G[1][2] == {}