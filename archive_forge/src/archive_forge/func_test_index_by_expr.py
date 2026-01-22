import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_index_by_expr(self):
    m = ConcreteModel()
    m.act = IntervalVar([(1, 2), (2, 1), (2, 2)])
    m.i = Var(domain=Integers)
    m.i2 = Var([1, 2], domain=Integers)
    thing1 = m.act[m.i, 2]
    self.assertIsInstance(thing1, GetItemExpression)
    self.assertEqual(len(thing1.args), 3)
    self.assertIs(thing1.args[0], m.act)
    self.assertIs(thing1.args[1], m.i)
    self.assertEqual(thing1.args[2], 2)
    thing2 = thing1.start_time
    self.assertIsInstance(thing2, GetAttrExpression)
    self.assertEqual(len(thing2.args), 2)
    self.assertIs(thing2.args[0], thing1)
    self.assertEqual(thing2.args[1], 'start_time')
    expr1 = m.act[m.i, 2].start_time.before(m.act[m.i ** 2, 1].end_time)