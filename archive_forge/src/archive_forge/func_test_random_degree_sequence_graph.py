import pytest
import networkx as nx
def test_random_degree_sequence_graph():
    d = [1, 2, 2, 3]
    G = nx.random_degree_sequence_graph(d, seed=42)
    assert d == sorted((d for n, d in G.degree()))