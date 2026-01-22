import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
def test_range_relational(self):
    a = RP([[NR(0, 10, 1)], [NR(0, 10, 0), NNR('a')]])
    aa = RP([[NR(0, 10, 1)], [NR(0, 10, 0), NNR('a')]])
    b = RP([[NR(0, 10, 1)], [NR(0, 10, 0), NNR('a'), NNR('b')]])
    c = RP([[NR(0, 10, 1)], [NR(0, 10, 0), NNR('b')]])
    d = RP([[NR(0, 10, 0)], [NR(0, 10, 0), NNR('a')]])
    d = RP([[NR(0, 10, 0)], [AnyRange()]])
    self.assertTrue(a.issubset(aa))
    self.assertTrue(a.issubset(b))
    self.assertFalse(a.issubset(c))
    self.assertTrue(a.issubset(d))
    self.assertFalse(a.issubset(NNR('a')))
    self.assertFalse(a.issubset(NR(None, None, 0)))
    self.assertTrue(a.issubset(AnyRange()))