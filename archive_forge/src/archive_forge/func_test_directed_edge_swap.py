import pytest
import networkx as nx
def test_directed_edge_swap():
    graph = nx.path_graph(200, create_using=nx.DiGraph)
    in_degrees = sorted(((n, d) for n, d in graph.in_degree()))
    out_degrees = sorted(((n, d) for n, d in graph.out_degree()))
    G = nx.directed_edge_swap(graph, nswap=40, max_tries=500, seed=1)
    assert in_degrees == sorted(((n, d) for n, d in G.in_degree()))
    assert out_degrees == sorted(((n, d) for n, d in G.out_degree()))