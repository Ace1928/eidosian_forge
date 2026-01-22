import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_deactivate_trivial_constraints_revert(self):
    """Test for reversion of trivial constraint deactivation."""
    m = ConcreteModel()
    m.v1 = Var(initialize=1)
    m.v2 = Var(initialize=2)
    m.v3 = Var(initialize=3)
    m.c = Constraint(expr=m.v1 <= m.v2)
    m.c2 = Constraint(expr=m.v2 >= m.v3)
    m.c3 = Constraint(expr=m.v1 <= 5)
    m.v1.fix()
    xfrm = TransformationFactory('contrib.deactivate_trivial_constraints')
    xfrm.apply_to(m, tmp=True)
    self.assertTrue(m.c.active)
    self.assertTrue(m.c2.active)
    self.assertFalse(m.c3.active)
    xfrm.revert(m)
    self.assertTrue(m.c3.active)