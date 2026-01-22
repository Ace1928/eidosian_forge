import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_loop(self):
    e = [(1, 2), (2, 2)]
    G = nx.Graph(e)
    self.check_graph(G, is_planar=True)