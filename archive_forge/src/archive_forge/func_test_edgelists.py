import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edgelists(self):
    P = nx.path_graph(4)
    e = [(0, 1), (1, 2), (2, 3)]
    G = nx.Graph(e)
    assert nodes_equal(sorted(G.nodes()), sorted(P.nodes()))
    assert edges_equal(sorted(G.edges()), sorted(P.edges()))
    assert edges_equal(sorted(G.edges(data=True)), sorted(P.edges(data=True)))
    e = [(0, 1, {}), (1, 2, {}), (2, 3, {})]
    G = nx.Graph(e)
    assert nodes_equal(sorted(G.nodes()), sorted(P.nodes()))
    assert edges_equal(sorted(G.edges()), sorted(P.edges()))
    assert edges_equal(sorted(G.edges(data=True)), sorted(P.edges(data=True)))
    e = ((n, n + 1) for n in range(3))
    G = nx.Graph(e)
    assert nodes_equal(sorted(G.nodes()), sorted(P.nodes()))
    assert edges_equal(sorted(G.edges()), sorted(P.edges()))
    assert edges_equal(sorted(G.edges(data=True)), sorted(P.edges(data=True)))