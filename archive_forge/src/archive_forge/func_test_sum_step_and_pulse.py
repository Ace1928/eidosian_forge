import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_step_and_pulse(self):
    m = self.get_model()
    expr = Step(m.a.start_time, height=4) + Pulse((m.b, -1))
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 2)
    self.assertEqual(len(expr.args), 2)
    self.assertIsInstance(expr.args[0], StepAtStart)
    self.assertIsInstance(expr.args[1], Pulse)
    self.assertEqual(str(expr), 'Step(a.start_time, height=4) + Pulse(b, height=-1)')