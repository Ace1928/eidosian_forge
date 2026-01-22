import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_multiple_components_planar(self):
    e = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
    self.check_graph(nx.Graph(e), is_planar=True)