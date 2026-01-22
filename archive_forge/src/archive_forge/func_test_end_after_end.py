import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_end_after_end(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.a.end_time.after(m.b.end_time))
    self.assertIsInstance(m.c.expr, BeforeExpression)
    self.assertEqual(len(m.c.expr.args), 3)
    self.assertIs(m.c.expr.args[0], m.b.end_time)
    self.assertIs(m.c.expr.args[1], m.a.end_time)
    self.assertEqual(m.c.expr.delay, 0)
    self.assertEqual(str(m.c.expr), 'b.end_time <= a.end_time')