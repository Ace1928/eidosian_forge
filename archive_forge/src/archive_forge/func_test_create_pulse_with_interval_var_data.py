import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_create_pulse_with_interval_var_data(self):
    m = self.get_model()
    p = Pulse(interval_var=m.c[2], height=2)
    self.assertIsInstance(p, Pulse)
    self.assertEqual(str(p), 'Pulse(c[2], height=2)')