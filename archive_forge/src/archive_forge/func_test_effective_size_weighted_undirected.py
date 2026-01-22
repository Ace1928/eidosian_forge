import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_weighted_undirected(self):
    G = self.G.copy()
    nx.set_edge_attributes(G, self.G_weights, 'weight')
    effective_size = nx.effective_size(G, weight='weight')
    assert effective_size['G'] == pytest.approx(5.47, abs=0.01)
    assert effective_size['A'] == pytest.approx(2.47, abs=0.01)
    assert effective_size['C'] == pytest.approx(1, abs=0.01)