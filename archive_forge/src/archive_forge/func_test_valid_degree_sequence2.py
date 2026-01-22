import pytest
import networkx as nx
def test_valid_degree_sequence2():
    n = 100
    for i in range(10):
        G = nx.barabasi_albert_graph(n, 1)
        deg = (d for n, d in G.degree())
        assert nx.is_graphical(deg, method='eg')
        assert nx.is_graphical(deg, method='hh')