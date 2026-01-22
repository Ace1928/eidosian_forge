import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
def test_trivial_constraints_ub_conflict(self):
    """Test for violated trivial constraint upper bound."""
    with self.assertRaisesRegex(InfeasibleConstraintException, 'Trivial constraint c violates BODY 1 ≤ UB 0.0.'):
        self._trivial_constraints_ub_conflict()