import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
def test_info_methods(self):
    a = NNR('a')
    b = NR(0, 5, 0)
    c = NR(5, 10, 1)
    x = RP([[a], [b, c]])
    y = RP([[a], [c]])
    self.assertFalse(x.isdiscrete())
    self.assertFalse(x.isfinite())
    self.assertTrue(y.isdiscrete())
    self.assertTrue(y.isfinite())