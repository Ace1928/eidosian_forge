import pytest
import networkx as nx
def test_domrel_png(self):
    edges = [(1, 2), (2, 3), (2, 4), (2, 6), (3, 5), (4, 5), (5, 2)]
    G = nx.DiGraph(edges)
    assert nx.dominance_frontiers(G, 1) == {1: set(), 2: {2}, 3: {5}, 4: {5}, 5: {2}, 6: set()}
    result = nx.dominance_frontiers(G.reverse(copy=False), 6)
    assert result == {1: set(), 2: {2}, 3: {2}, 4: {2}, 5: {2}, 6: set()}