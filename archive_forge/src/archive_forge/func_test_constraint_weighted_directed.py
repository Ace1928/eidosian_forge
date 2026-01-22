import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_constraint_weighted_directed(self):
    D = self.D.copy()
    nx.set_edge_attributes(D, self.D_weights, 'weight')
    constraint = nx.constraint(D, weight='weight')
    assert constraint[0] == pytest.approx(0.84, abs=0.001)
    assert constraint[1] == pytest.approx(1.143, abs=0.001)
    assert constraint[2] == pytest.approx(1.378, abs=0.001)