import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_k3_3(self):
    self.check_graph(nx.complete_bipartite_graph(3, 3), is_planar=False)