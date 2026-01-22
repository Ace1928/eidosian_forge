from itertools import combinations
import pytest
import networkx as nx
def test_isolated_K5():
    G = nx.Graph()
    G.add_edges_from(combinations(range(5), 2))
    G.add_edges_from(combinations(range(5, 10), 2))
    c = set(nx.community.k_clique_communities(G, 5))
    assert c == {frozenset(range(5)), frozenset(range(5, 10))}