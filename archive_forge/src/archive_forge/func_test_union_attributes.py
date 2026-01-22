import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_union_attributes():
    g = nx.Graph()
    g.add_node(0, x=4)
    g.add_node(1, x=5)
    g.add_edge(0, 1, size=5)
    g.graph['name'] = 'g'
    h = g.copy()
    h.graph['name'] = 'h'
    h.graph['attr'] = 'attr'
    h.nodes[0]['x'] = 7
    gh = nx.union(g, h, rename=('g', 'h'))
    assert set(gh.nodes()) == {'h0', 'h1', 'g0', 'g1'}
    for n in gh:
        graph, node = n
        assert gh.nodes[n] == eval(graph).nodes[int(node)]
    assert gh.graph['attr'] == 'attr'
    assert gh.graph['name'] == 'h'