import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_non_planar1(self):
    e = [(1, 5), (1, 6), (1, 7), (2, 6), (2, 3), (3, 5), (3, 7), (4, 5), (4, 6), (4, 7)]
    self.check_graph(nx.Graph(e), is_planar=False)