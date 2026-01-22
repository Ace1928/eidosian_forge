import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_assortativity(self):
    a = np.array([[50, 50, 0], [50, 50, 0], [0, 0, 2]])
    r = attribute_ac(a)
    np.testing.assert_almost_equal(r, 0.029, decimal=3)