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
def test_nonl(self):
    m = ConcreteModel()
    m.w = Var(initialize=0)
    e = 1 + sin(m.w)
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '1 + sin(w)')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '1 + sin(w)')
    m.w.fixed = True
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '1.0')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '1 + sin(w)')