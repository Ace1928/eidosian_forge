import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_detect_fixed_false(self):
    m = ConcreteModel()
    m.x = Var()
    m.c = Constraint(expr=m.x == 3)
    TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m, detect_fixed=False)
    self.assertFalse(m.c.active)
    self.assertTrue(m.x.has_lb())
    self.assertEqual(m.x.lb, 3)
    self.assertTrue(m.x.has_ub())
    self.assertEqual(m.x.ub, 3)
    self.assertFalse(m.x.fixed)