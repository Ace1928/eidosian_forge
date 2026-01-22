import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_smaller_right(self):
    G = nx.complete_bipartite_graph(4, 3)
    G.add_edge(0, 4, weight=400)
    G.add_edge(0, 5, weight=400)
    G.add_edge(0, 6, weight=300)
    G.add_edge(1, 4, weight=150)
    G.add_edge(1, 5, weight=450)
    G.add_edge(1, 6, weight=225)
    G.add_edge(2, 4, weight=400)
    G.add_edge(2, 5, weight=600)
    G.add_edge(2, 6, weight=290)
    G.add_edge(3, 4, weight=1)
    G.add_edge(3, 5, weight=2)
    G.add_edge(3, 6, weight=3)
    matching = minimum_weight_full_matching(G)
    assert matching == {1: 4, 2: 6, 3: 5, 4: 1, 5: 3, 6: 2}