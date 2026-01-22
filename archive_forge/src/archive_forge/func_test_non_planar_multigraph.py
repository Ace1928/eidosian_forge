import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_non_planar_multigraph(self):
    G = nx.MultiGraph(nx.complete_graph(5))
    G.add_edges_from([(1, 2)] * 5)
    self.check_graph(G, is_planar=False)