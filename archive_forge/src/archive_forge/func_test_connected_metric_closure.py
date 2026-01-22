import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
def test_connected_metric_closure(self):
    G = self.G1.copy()
    G.add_node(100)
    pytest.raises(nx.NetworkXError, metric_closure, G)