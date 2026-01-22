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
def test_pow4(self):
    m = ConcreteModel()
    m.a = Var(initialize=2)
    m.b = Var(initialize=0)
    m.a.fixed = True
    m.b.fixed = True
    e = m.a ** m.b
    rep = generate_standard_repn(e, compute_values=False, quadratic=False)
    self.assertTrue(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 0)
    self.assertTrue(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    rep = generate_standard_repn(e, compute_values=True, quadratic=False)
    self.assertTrue(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 0)
    self.assertTrue(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    m.b.fixed = False
    rep = generate_standard_repn(e, compute_values=False, quadratic=False)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), None)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertFalse(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 1)
    baseline = {}
    self.assertEqual(baseline, repn_to_dict(rep))