import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_smaller_left(self):
    G = nx.complete_bipartite_graph(3, 4)
    G.add_edge(0, 3, weight=400)
    G.add_edge(0, 4, weight=150)
    G.add_edge(0, 5, weight=400)
    G.add_edge(0, 6, weight=1)
    G.add_edge(1, 3, weight=400)
    G.add_edge(1, 4, weight=450)
    G.add_edge(1, 5, weight=600)
    G.add_edge(1, 6, weight=2)
    G.add_edge(2, 3, weight=300)
    G.add_edge(2, 4, weight=225)
    G.add_edge(2, 5, weight=290)
    G.add_edge(2, 6, weight=3)
    matching = minimum_weight_full_matching(G)
    assert matching == {0: 4, 1: 6, 2: 5, 4: 0, 5: 2, 6: 1}