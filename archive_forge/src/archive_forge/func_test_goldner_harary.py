import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_goldner_harary(self):
    e = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 10), (1, 11), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10), (2, 11), (3, 4), (4, 5), (4, 6), (4, 7), (5, 7), (6, 7), (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (10, 11)]
    G = nx.Graph(e)
    self.check_graph(G, is_planar=True)