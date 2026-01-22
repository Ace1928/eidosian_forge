import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_intersection_all_attributes_different_node_sets():
    g = nx.Graph()
    g.add_node(0, x=4)
    g.add_node(1, x=5)
    g.add_edge(0, 1, size=5)
    g.graph['name'] = 'g'
    h = g.copy()
    g.add_node(2)
    h.graph['name'] = 'h'
    h.graph['attr'] = 'attr'
    h.nodes[0]['x'] = 7
    gh = nx.intersection_all([g, h])
    assert set(gh.nodes()) == set(h.nodes())
    assert sorted(gh.edges()) == sorted(g.edges())