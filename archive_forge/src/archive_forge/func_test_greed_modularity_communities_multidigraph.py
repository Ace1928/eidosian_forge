import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greed_modularity_communities_multidigraph():
    G = nx.MultiDiGraph()
    G.add_edges_from([(1, 2), (1, 2), (3, 1), (2, 3), (2, 3), (3, 2), (1, 4), (2, 4), (4, 2), (4, 5), (5, 6), (5, 6), (6, 5), (5, 7), (6, 7), (7, 8), (5, 8), (8, 4)])
    expected = [frozenset({1, 2, 3, 4}), frozenset({5, 6, 7, 8})]
    assert greedy_modularity_communities(G, weight='weight') == expected