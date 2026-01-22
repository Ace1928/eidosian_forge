import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_comp(self):
    e = [(1, 2), (3, 4)]
    G = nx.Graph(e)
    G.remove_edge(1, 2)
    self.check_graph(G, is_planar=True)