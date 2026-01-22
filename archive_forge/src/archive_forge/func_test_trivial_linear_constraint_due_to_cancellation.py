import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_trivial_linear_constraint_due_to_cancellation(self):
    m = ConcreteModel()
    m.x = Var()
    m.c = Constraint(expr=m.x - m.x <= 0)
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
    self.assertFalse(m.c.active)