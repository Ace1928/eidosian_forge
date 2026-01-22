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
def test_linear_with_mutable_param_and_fixed_var(self):
    m = ConcreteModel()
    m.A = RangeSet(5)
    m.v = Var(m.A, initialize=1)
    m.p = Param(m.A, initialize={1: -2, 2: -1, 3: 0, 4: 1, 5: 2}, mutable=True)
    with EXPR.linear_expression() as expr:
        for i in m.A:
            expr += m.p[i] * m.v[i]
    e = summation(m.v) + expr
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '- v[1] + v[3] + 2*v[4] + 3*v[5]')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), '(1 + p[1])*v[1] + (1 + p[2])*v[2] + (1 + p[3])*v[3] + (1 + p[4])*v[4] + (1 + p[5])*v[5]')
    m.v[1].fixed = True
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '-1 + v[3] + 2*v[4] + 3*v[5]')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'v[1] + p[1]*v[1] + (1 + p[2])*v[2] + (1 + p[3])*v[3] + (1 + p[4])*v[4] + (1 + p[5])*v[5]')