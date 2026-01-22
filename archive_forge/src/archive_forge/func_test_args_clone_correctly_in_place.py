import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_args_clone_correctly_in_place(self):
    m = self.get_model()
    m.p1 = Pulse(interval_var=m.a, height=3)
    m.p2 = Pulse(interval_var=m.b, height=4)
    m.s = Step(m.a.start_time, height=-1)
    expr1 = m.p1 - m.p2
    expr = expr1 + m.p1
    expr1 -= m.s
    self.assertIsInstance(expr1, CumulativeFunction)
    self.assertEqual(expr1.nargs(), 3)
    self.assertIs(expr1.args[0], m.p1)
    self.assertIsInstance(expr1.args[1], NegatedStepFunction)
    self.assertIs(expr1.args[1].args[0], m.p2)
    self.assertIsInstance(expr1.args[2], NegatedStepFunction)
    self.assertIs(expr1.args[2].args[0], m.s)
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 3)
    self.assertIs(expr.args[0], m.p1)
    self.assertIsInstance(expr.args[1], NegatedStepFunction)
    self.assertIs(expr.args[1].args[0], m.p2)
    self.assertIs(expr.args[2], m.p1)