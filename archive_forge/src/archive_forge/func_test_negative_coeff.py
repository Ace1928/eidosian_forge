import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Constraint, TransformationFactory, Var, value
def test_negative_coeff(self):
    """Unbounded in one direction with negative coefficient"""
    m = ConcreteModel()
    m.v1 = Var(initialize=7, bounds=(1, float('inf')))
    m.v2 = Var(initialize=2, bounds=(2, 5))
    m.v3 = Var(initialize=6, bounds=(6, 9))
    m.v4 = Var(initialize=1, bounds=(1, 1))
    m.c1 = Constraint(expr=2 * m.v2 + m.v3 + m.v4 - m.v1 <= 50)
    self.assertEqual(value(m.c1.upper), 50)
    self.assertTrue(m.c1.has_ub())
    self.assertFalse(m.c1.has_lb())
    TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
    self.assertEqual(value(m.c1.upper), 19)
    self.assertFalse(m.c1.has_lb())