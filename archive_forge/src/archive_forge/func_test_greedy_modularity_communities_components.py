import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_greedy_modularity_communities_components():
    G = nx.Graph([(0, 1), (2, 3), (4, 5), (5, 6)])
    assert greedy_modularity_communities(G) == [{4, 5, 6}, {0, 1}, {2, 3}]
    assert greedy_modularity_communities(G, best_n=3) == [{4, 5, 6}, {0, 1}, {2, 3}]
    assert greedy_modularity_communities(G, best_n=2) == [{0, 1, 4, 5, 6}, {2, 3}]
    assert greedy_modularity_communities(G, best_n=1) == [{0, 1, 2, 3, 4, 5, 6}]