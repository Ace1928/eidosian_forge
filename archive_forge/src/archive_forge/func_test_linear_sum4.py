import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def test_linear_sum4(self):
    m = ConcreteModel()
    m.A = Set(initialize=range(5))
    m.x = Var(m.A, initialize=3)
    m.p = Param(m.A, mutable=True, default=1)
    e = quicksum((m.p[i] * m.x[i] for i in m.A))
    m.x[2].fixed = True
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 4)
    self.assertTrue(len(rep.linear_coefs) == 4)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.x[0]): 1, id(m.x[1]): 1, None: 3, id(m.x[3]): 1, id(m.x[4]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    rep = generate_standard_repn(e, compute_values=False)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 4)
    self.assertTrue(len(rep.linear_coefs) == 4)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.x[0]): 1, id(m.x[1]): 1, None: 3, id(m.x[3]): 1, id(m.x[4]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    self.assertTrue(rep.linear_coefs[0] is m.p[0])
    self.assertTrue(rep.linear_coefs[1] is m.p[1])
    self.assertTrue(type(rep.constant) is EXPR.MonomialTermExpression)