import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_subtract_steps_in_place(self):
    m = self.get_model()
    s1 = Step(m.a.start_time, height=1)
    s2 = Step(m.b.end_time, height=3)
    expr = s1
    expr -= s2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(len(expr.args), 2)
    self.assertEqual(expr.nargs(), 2)
    self.assertIs(expr.args[0], s1)
    self.assertIsInstance(expr.args[1], NegatedStepFunction)
    self.assertIs(expr.args[1].args[0], s2)