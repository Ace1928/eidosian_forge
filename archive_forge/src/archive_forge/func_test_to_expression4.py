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
def test_to_expression4(self):
    m = ConcreteModel()
    m.A = RangeSet(3)
    m.v = Var(m.A)
    m.p = Param(m.A, initialize={1: -1, 2: 0, 3: 1}, mutable=True)
    e = sum((m.p[i] * m.v[i] ** 2 for i in m.v))
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'p[1]*v[1]**2 + p[2]*v[2]**2 + p[3]*v[3]**2')
    e = sin(m.v[1])
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'sin(v[1])')