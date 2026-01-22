import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_skip_trivial_constraints(self):
    """Tests handling of zero coefficients."""
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.c = Constraint(expr=m.x * m.y == m.z)
    m.z.fix(0)
    m.y.fix(0)
    TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
    self.assertTrue(m.c.active)
    self.assertFalse(m.x.has_lb())
    self.assertFalse(m.x.has_ub())