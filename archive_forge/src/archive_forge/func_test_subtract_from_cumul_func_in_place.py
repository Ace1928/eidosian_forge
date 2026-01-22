import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_subtract_from_cumul_func_in_place(self):
    m = self.get_model()
    m.p1 = Pulse(interval_var=m.a, height=5)
    m.p2 = Pulse(interval_var=m.b, height=-3)
    m.s = Step(m.b.end_time, height=5)
    expr = m.p1 + m.s
    expr -= m.p2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 3)
    self.assertIs(expr.args[0], m.p1)
    self.assertIs(expr.args[1], m.s)
    self.assertIsInstance(expr.args[2], NegatedStepFunction)
    self.assertIs(expr.args[2].args[0], m.p2)
    self.assertEqual(str(expr), 'Pulse(a, height=5) + Step(b.end_time, height=5) - Pulse(b, height=-3)')