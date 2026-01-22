import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_constant_length_and_start(self):
    m = ConcreteModel()
    m.i = IntervalVar(length=7, start=3)
    self.assertEqual(m.i.length.lower, 7)
    self.assertEqual(m.i.length.upper, 7)
    self.assertEqual(m.i.start_time.lower, 3)
    self.assertEqual(m.i.start_time.upper, 3)