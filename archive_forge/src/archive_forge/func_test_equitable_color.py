import itertools
import pytest
import networkx as nx
def test_equitable_color(self):
    G = nx.fast_gnp_random_graph(n=10, p=0.2, seed=42)
    coloring = nx.coloring.equitable_color(G, max_degree(G) + 1)
    assert is_equitable(G, coloring)