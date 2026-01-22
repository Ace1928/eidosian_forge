import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
def test_local_var(self):
    m = models.localVar()
    binary_multiplication = TransformationFactory('gdp.binary_multiplication')
    binary_multiplication.apply_to(m)
    transformedC = binary_multiplication.get_transformed_constraints(m.disj2.cons)
    self.assertEqual(len(transformedC), 1)
    eq = transformedC[0]
    repn = generate_standard_repn(eq.body)
    self.assertIsNone(repn.nonlinear_expr)
    self.assertEqual(len(repn.linear_coefs), 1)
    self.assertEqual(len(repn.quadratic_coefs), 2)
    ct.check_linear_coef(self, repn, m.disj2.indicator_var, -3)
    ct.check_quadratic_coef(self, repn, m.x, m.disj2.indicator_var, 1)
    ct.check_quadratic_coef(self, repn, m.disj2.y, m.disj2.indicator_var, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(eq.lb, 0)
    self.assertEqual(eq.ub, 0)