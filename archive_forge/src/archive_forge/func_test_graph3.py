import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_graph3(self):
    G = nx.Graph([(0, 7), (3, 11), (3, 4), (8, 9), (4, 11), (1, 7), (1, 13), (1, 11), (3, 5), (5, 7), (1, 3), (0, 4), (5, 11), (5, 13)])
    self.check_graph(G, is_planar=False)