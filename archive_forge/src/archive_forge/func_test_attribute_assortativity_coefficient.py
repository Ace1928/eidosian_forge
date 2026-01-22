import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_assortativity_coefficient(self):
    a = np.array([[0.258, 0.016, 0.035, 0.013], [0.012, 0.157, 0.058, 0.019], [0.013, 0.023, 0.306, 0.035], [0.005, 0.007, 0.024, 0.016]])
    r = attribute_ac(a)
    np.testing.assert_almost_equal(r, 0.623, decimal=3)