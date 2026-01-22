import pyomo.common.unittest as unittest
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.environ import (
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunction
def check_lor_on_disjunct(self, model, disjunct, x1, x2):
    x1 = x1.get_associated_binary()
    x2 = x2.get_associated_binary()
    disj0 = disjunct.logic_to_linear
    self.assertEqual(len(disj0.component_map(Constraint)), 1)
    lor = disj0.transformed_constraints[1]
    self.assertEqual(lor.lower, 1)
    self.assertIsNone(lor.upper)
    repn = generate_standard_repn(lor.body)
    self.assertEqual(repn.constant, 0)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertIs(repn.linear_vars[0], x1)
    self.assertIs(repn.linear_vars[1], x2)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertEqual(repn.linear_coefs[1], 1)