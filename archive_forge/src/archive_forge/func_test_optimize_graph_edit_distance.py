import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_optimize_graph_edit_distance(self):
    G1 = circular_ladder_graph(2)
    G2 = circular_ladder_graph(6)
    bestcost = 1000
    for cost in optimize_graph_edit_distance(G1, G2):
        assert cost < bestcost
        bestcost = cost
    assert bestcost == 22