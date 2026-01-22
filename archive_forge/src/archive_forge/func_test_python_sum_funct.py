import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_python_sum_funct(self):
    m = self.get_model()
    expr = sum((Pulse(interval_var=m.c[i], height=1) for i in [1, 2]))
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(len(expr.args), 2)
    self.assertEqual(expr.nargs(), 2)
    self.assertIsInstance(expr.args[0], Pulse)
    self.assertIsInstance(expr.args[1], Pulse)