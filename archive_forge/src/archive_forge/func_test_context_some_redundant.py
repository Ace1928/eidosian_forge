import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def test_context_some_redundant(self):
    m = _make_simple_model()
    to_fix = [m.v2, m.v4]
    to_deactivate = [m.con1, m.con2]
    to_reset = [m.v1]
    m.v1.set_value(1.5)
    m.v2.fix()
    m.con1.deactivate()
    with TemporarySubsystemManager(to_fix, to_deactivate, to_reset):
        self.assertEqual(m.v1.value, 1.5)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertFalse(m.con1.active)
        self.assertFalse(m.con2.active)
        m.v1.set_value(2.0)
        m.v2.set_value(3.0)
    self.assertEqual(m.v1.value, 1.5)
    self.assertEqual(m.v2.value, 3.0)
    self.assertTrue(m.v2.fixed)
    self.assertFalse(m.v4.fixed)
    self.assertTrue(m.con2.active)
    self.assertFalse(m.con1.active)