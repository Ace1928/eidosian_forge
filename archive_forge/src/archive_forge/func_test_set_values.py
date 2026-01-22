import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def test_set_values(self):
    m = _make_simple_model()
    n_scenario = 2
    input_values = ComponentMap([(m.v3, [1.3, 2.3]), (m.v4, [1.4, 2.4])])
    to_fix = [m.v3, m.v4]
    to_deactivate = [m.con1]
    with ParamSweeper(2, input_values, to_fix=to_fix, to_deactivate=to_deactivate) as sweeper:
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertFalse(m.con1.active)
        self.assertTrue(m.con2.active)
        self.assertTrue(m.con3.active)
        for i, (inputs, outputs) in enumerate(sweeper):
            self.assertEqual(len(inputs), 2)
            self.assertEqual(len(outputs), 0)
            self.assertIn(m.v3, inputs)
            self.assertIn(m.v4, inputs)
            for var, val in inputs.items():
                self.assertEqual(var.value, val)
                self.assertEqual(var.value, input_values[var][i])
    self.assertIs(m.v3.value, None)
    self.assertIs(m.v4.value, None)