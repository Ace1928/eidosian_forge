import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_weight(self):
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    G.add_edge(1, 2, weight=-3)
    assert nx.single_source_bellman_ford_path(G, 0) == {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4]}
    assert nx.single_source_bellman_ford_path_length(G, 0) == {0: 0, 1: 1, 2: -2, 3: -1, 4: 0}
    assert nx.single_source_bellman_ford(G, 0) == ({0: 0, 1: 1, 2: -2, 3: -1, 4: 0}, {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4]})
    assert nx.bellman_ford_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, {0: 0, 1: 1, 2: -2, 3: -1, 4: 0})
    assert nx.goldberg_radzik(G, 0) == ({0: None, 1: 0, 2: 1, 3: 2, 4: 3}, {0: 0, 1: 1, 2: -2, 3: -1, 4: 0})