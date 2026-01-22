import pytest
import networkx as nx
def test_degree_seq_c4():
    G = nx.cycle_graph(4)
    degrees = sorted((d for n, d in G.degree()))
    G = nx.double_edge_swap(G, 1, 100)
    assert degrees == sorted((d for n, d in G.degree()))