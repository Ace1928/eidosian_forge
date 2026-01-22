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
def test_fixed_exponent(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    e = m.y + 2 ** m.x
    m.x.fix(1)
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 1)
    self.assertTrue(len(rep.linear_coefs) == 1)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.y): 1, None: 2}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 1, None: 2}
    self.assertEqual(baseline, repn_to_dict(rep))