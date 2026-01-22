import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_constraint_isolated(self):
    G = self.G.copy()
    G.add_node(1)
    constraint = nx.constraint(G)
    assert math.isnan(constraint[1])