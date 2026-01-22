import pytest
import networkx as nx
def test_valid_degree_sequence1():
    n = 100
    p = 0.3
    for i in range(10):
        G = nx.erdos_renyi_graph(n, p)
        deg = (d for n, d in G.degree())
        assert nx.is_graphical(deg, method='eg')
        assert nx.is_graphical(deg, method='hh')