import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import (
def test_var_bound_propagate_crossover(self):
    """Test for error message when variable bound crosses over."""
    m = ConcreteModel()
    m.v1 = Var(initialize=1, bounds=(1, 3))
    m.v2 = Var(initialize=5, bounds=(4, 8))
    m.c1 = Constraint(expr=m.v1 == m.v2)
    xfrm = TransformationFactory('contrib.propagate_eq_var_bounds')
    with self.assertRaisesRegex(InfeasibleConstraintException, 'Variable v2 has a lower bound 4 > the upper bound 3 of variable v1, but they are linked by equality constraints'):
        xfrm.apply_to(m)