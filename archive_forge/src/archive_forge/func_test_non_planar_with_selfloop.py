import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_non_planar_with_selfloop(self):
    G = nx.complete_graph(5)
    for i in range(5):
        G.add_edge(i, i)
    self.check_graph(G, is_planar=False)