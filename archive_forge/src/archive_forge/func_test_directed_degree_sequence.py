import pytest
import networkx as nx
def test_directed_degree_sequence():
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(r):
        G = nx.erdos_renyi_graph(n, p * (i + 1), None, True)
        din = (d for n, d in G.in_degree())
        dout = (d for n, d in G.out_degree())
        assert nx.is_digraphical(din, dout)