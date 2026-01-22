import pytest
import networkx as nx
def test_directed_edge_swap_undo_previous_swap():
    G = nx.DiGraph(nx.path_graph(4).edges)
    edges = set(G.edges)
    nx.directed_edge_swap(G, nswap=2, max_tries=100)
    assert edges == set(G.edges)
    nx.directed_edge_swap(G, nswap=1, max_tries=100, seed=1)
    assert {(0, 2), (1, 3), (2, 1)} == set(G.edges)
    nx.directed_edge_swap(G, nswap=1, max_tries=100, seed=1)
    assert edges == set(G.edges)