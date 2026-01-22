import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_multigraph():
    G = nx.MultiGraph()
    G.add_edges_from([(1, 2), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (4, 5), (5, 6), (5, 7), (5, 7), (6, 7), (7, 8), (5, 8)])
    expected = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    assert greedy_modularity_communities(G) == expected
    G.add_edge(4, 5)
    expected = [frozenset({4, 5, 6, 7, 8}), frozenset({1, 2, 3})]
    assert greedy_modularity_communities(G) == expected