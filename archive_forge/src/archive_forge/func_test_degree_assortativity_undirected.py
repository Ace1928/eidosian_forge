import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_assortativity_undirected(self):
    r = nx.degree_assortativity_coefficient(self.P4)
    np.testing.assert_almost_equal(r, -1.0 / 2, decimal=4)