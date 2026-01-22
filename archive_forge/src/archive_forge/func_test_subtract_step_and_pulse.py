import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_subtract_step_and_pulse(self):
    m = self.get_model()
    s1 = Step(m.a.end_time, height=2)
    s2 = Step(m.b.start_time, height=5)
    p = Pulse(interval_var=m.a, height=3)
    expr = s1 - s2 - p
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(len(expr.args), 3)
    self.assertEqual(expr.nargs(), 3)
    self.assertIs(expr.args[0], s1)
    self.assertIsInstance(expr.args[1], NegatedStepFunction)
    self.assertIs(expr.args[1].args[0], s2)
    self.assertIsInstance(expr.args[2], NegatedStepFunction)
    self.assertIs(expr.args[2].args[0], p)