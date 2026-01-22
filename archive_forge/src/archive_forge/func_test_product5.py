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
def test_product5(self):
    m = ConcreteModel()
    m.v = Var(initialize=2)
    m.w = Var(initialize=3)
    e = (1 + m.v) * (1 + m.v)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '1 + 2*v + v**2')
    rep = generate_standard_repn(e, compute_values=True, quadratic=False)
    self.assertEqual(str(rep.to_expression()), '(1 + v)*(1 + v)')