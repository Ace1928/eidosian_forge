import pytest
import networkx as nx
def test_small_graph_centrality(self):
    G = nx.empty_graph(create_using=nx.DiGraph)
    assert {} == nx.degree_centrality(G)
    assert {} == nx.out_degree_centrality(G)
    assert {} == nx.in_degree_centrality(G)
    G = nx.empty_graph(1, create_using=nx.DiGraph)
    assert {0: 1} == nx.degree_centrality(G)
    assert {0: 1} == nx.out_degree_centrality(G)
    assert {0: 1} == nx.in_degree_centrality(G)