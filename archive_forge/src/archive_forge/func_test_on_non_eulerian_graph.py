import collections
import pytest
import networkx as nx
def test_on_non_eulerian_graph(self):
    G = nx.cycle_graph(18)
    G.add_edge(0, 18)
    G.add_edge(18, 19)
    G.add_edge(17, 19)
    G.add_edge(4, 20)
    G.add_edge(20, 21)
    G.add_edge(21, 22)
    G.add_edge(22, 23)
    G.add_edge(23, 24)
    G.add_edge(24, 25)
    G.add_edge(25, 26)
    G.add_edge(26, 27)
    G.add_edge(27, 28)
    G.add_edge(28, 13)
    assert not nx.is_eulerian(G)
    G = nx.eulerize(G)
    assert nx.is_eulerian(G)
    assert nx.number_of_edges(G) == 39