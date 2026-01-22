import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_simple_planar_graph(self):
    e = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 1), (1, 5), (5, 2), (2, 4), (4, 5), (5, 7)]
    self.check_graph(nx.Graph(e), is_planar=True)