import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_two_cumul_funcs(self):
    m = self.get_model()
    s1 = Step(m.a.start_time, height=4)
    p1 = Step(m.a.start_time, height=4)
    cumul1 = s1 + p1
    s2 = Step(m.a.end_time, height=3)
    s3 = Step(0, height=34)
    cumul2 = s2 + s3
    expr = cumul1 + cumul2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 4)
    self.assertIs(expr.args[0], s1)
    self.assertIs(expr.args[1], p1)
    self.assertIs(expr.args[2], s2)
    self.assertIs(expr.args[3], s3)