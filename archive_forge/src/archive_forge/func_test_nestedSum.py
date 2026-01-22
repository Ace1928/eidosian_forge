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
def test_nestedSum(self):
    m = ConcreteModel()
    m.a = Var()
    m.b = Var()
    m.c = Var()
    m.d = Var()
    e1 = m.a + m.b
    e = e1 + 5
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 5, id(m.a): 1, id(m.b): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {None: 5, id(rep.linear_vars[0]): 1, id(rep.linear_vars[1]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e = 5 + e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 5, id(m.a): 1, id(m.b): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {None: 5, id(rep.linear_vars[0]): 1, id(rep.linear_vars[1]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e = e1 + m.c
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 3)
    self.assertTrue(len(rep.linear_coefs) == 3)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 1, id(m.b): 1, id(m.c): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 1, id(rep.linear_vars[1]): 1, id(rep.linear_vars[2]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e = m.c + e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 3)
    self.assertTrue(len(rep.linear_coefs) == 3)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 1, id(m.b): 1, id(m.c): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 1, id(rep.linear_vars[1]): 1, id(rep.linear_vars[2]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e2 = m.c + m.d
    e = e1 + e2
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
    baseline = {id(m.a): 1, id(m.b): 1, id(m.c): 1, id(m.d): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 1, id(rep.linear_vars[1]): 1, id(rep.linear_vars[2]): 1, id(rep.linear_vars[3]): 1}
    self.assertEqual(baseline, repn_to_dict(rep))