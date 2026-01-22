import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_counterexample_planar(self):
    with pytest.raises(nx.NetworkXException):
        G = nx.Graph()
        G.add_node(1)
        get_counterexample(G)