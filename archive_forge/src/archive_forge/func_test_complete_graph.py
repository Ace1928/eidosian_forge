import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_complete_graph(self):
    for m in [0, 1, 3, 5]:
        g = nx.complete_graph(m)
        assert nx.number_of_nodes(g) == m
        assert nx.number_of_edges(g) == m * (m - 1) // 2
    mg = nx.complete_graph(m, create_using=nx.MultiGraph)
    assert edges_equal(mg.edges(), g.edges())
    g = nx.complete_graph('abc')
    assert nodes_equal(g.nodes(), ['a', 'b', 'c'])
    assert g.size() == 3
    g = nx.complete_graph('abcb')
    assert nodes_equal(g.nodes(), ['a', 'b', 'c'])
    assert g.size() == 4
    g = nx.complete_graph('abcb', create_using=nx.MultiGraph)
    assert nodes_equal(g.nodes(), ['a', 'b', 'c'])
    assert g.size() == 6