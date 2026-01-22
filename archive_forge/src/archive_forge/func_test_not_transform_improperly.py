import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_not_transform_improperly(self):
    """Tests that invalid constraints are not transformed."""
    m = ConcreteModel()
    m.v1 = Var(initialize=0, domain=Binary)
    m.c1 = Constraint(expr=-1 * m.v1 <= 0)
    TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
    self.assertFalse(m.v1.fixed)