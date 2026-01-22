import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_isolated(self):
    G = self.G.copy()
    G.add_node(1)
    nx.set_edge_attributes(G, self.G_weights, 'weight')
    effective_size = nx.effective_size(G, weight='weight')
    assert math.isnan(effective_size[1])