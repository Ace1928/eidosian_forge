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
def test_nonlinear_sum(self):
    m = ConcreteModel()
    m.v = Var()
    e = 10 * (sin(m.v) + cos(m.v))
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '10*sin(v) + 10*cos(v)')
    e = 10 * (1 + sin(m.v))
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '10 + 10*sin(v)')