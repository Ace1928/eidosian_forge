import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_cutoff_parameter():
    G = nx.circular_ladder_graph(4)
    expected = [{k} for k in range(8)]
    assert greedy_modularity_communities(G, cutoff=8) == expected
    expected = [{k, k + 1} for k in range(0, 8, 2)]
    assert greedy_modularity_communities(G, cutoff=4) == expected
    expected = [frozenset(range(4)), frozenset(range(4, 8))]
    assert greedy_modularity_communities(G, cutoff=1) == expected