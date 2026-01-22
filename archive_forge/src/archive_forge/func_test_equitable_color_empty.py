import itertools
import pytest
import networkx as nx
def test_equitable_color_empty(self):
    G = nx.empty_graph()
    coloring = nx.coloring.equitable_color(G, max_degree(G) + 1)
    assert is_equitable(G, coloring)