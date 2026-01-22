import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_template_name(self):
    m = self.m
    t = IndexTemplate(m.I)
    E = m.x[t + m.P[1 + t]] + m.P[1]
    self.assertEqual(str(E), 'x[{I} + P[1 + {I}]] + P[1]')
    E = m.x[t + m.P[1 + t] ** 2.0] ** 2.0 + m.P[1]
    self.assertEqual(str(E), 'x[{I} + P[1 + {I}]**2.0]**2.0 + P[1]')