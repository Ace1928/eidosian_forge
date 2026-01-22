import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_initialize_with_no_data(self):
    m = ConcreteModel()
    m.i = IntervalVar([1, 2])
    for j in [1, 2]:
        self.assertIsInstance(m.i[j].start_time, IntervalVarTimePoint)
        self.assertEqual(m.i[j].start_time.domain, Integers)
        self.assertIsNone(m.i[j].start_time.lower)
        self.assertIsNone(m.i[j].start_time.upper)
        self.assertIsInstance(m.i[j].end_time, IntervalVarTimePoint)
        self.assertEqual(m.i[j].end_time.domain, Integers)
        self.assertIsNone(m.i[j].end_time.lower)
        self.assertIsNone(m.i[j].end_time.upper)
        self.assertIsInstance(m.i[j].length, IntervalVarLength)
        self.assertEqual(m.i[j].length.domain, Integers)
        self.assertIsNone(m.i[j].length.lower)
        self.assertIsNone(m.i[j].length.upper)
        self.assertIsInstance(m.i[j].is_present, IntervalVarPresence)