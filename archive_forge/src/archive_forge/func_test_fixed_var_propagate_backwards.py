import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import (
def test_fixed_var_propagate_backwards(self):
    """Test backwards propagation through equality set."""
    m = ConcreteModel()
    m.v1 = Var(initialize=1)
    m.v2 = Var(initialize=2)
    m.v3 = Var(initialize=3)
    m.v4 = Var(initialize=4)
    m.c1 = Constraint(expr=m.v1 == m.v2)
    m.c2 = Constraint(expr=m.v2 == m.v3)
    m.c3 = Constraint(expr=m.v3 == m.v4)
    m.v4.fix()
    TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
    self.assertTrue(m.v1.fixed)
    self.assertTrue(m.v2.fixed)
    self.assertTrue(m.v3.fixed)
    self.assertTrue(m.v4.fixed)
    self.assertEqual(value(m.v4), 4)