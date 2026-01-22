import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.filter import Filter, FilterElement
def test_isAcceptable(self):
    fe = FilterElement(0.5, 0.25)
    self.assertTrue(self.tmpFilter.isAcceptable(fe, self.theta_max))
    fe = FilterElement(10.0, 15.0)
    self.assertFalse(self.tmpFilter.isAcceptable(fe, self.theta_max))