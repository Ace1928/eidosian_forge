import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_cannot_subtract_constant(self):
    m = self.get_model()
    with self.assertRaisesRegex(TypeError, "Cannot subtract object of class <class 'int'> from object of class <class 'pyomo.contrib.cp.scheduling_expr.step_function_expressions.StepAtStart'>"):
        expr = Step(m.a.start_time, height=6) - 3