import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_single_component(self):
    G = nx.Graph()
    G.add_node(1)
    self.check_graph(G, is_planar=True)