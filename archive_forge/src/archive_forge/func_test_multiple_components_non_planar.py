import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_multiple_components_non_planar(self):
    G = nx.complete_graph(5)
    G.add_edges_from([(6, 7), (7, 8), (8, 6)])
    self.check_graph(G, is_planar=False)