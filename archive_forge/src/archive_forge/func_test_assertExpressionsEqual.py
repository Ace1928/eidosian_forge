from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.compare import (
from pyomo.common.gsl import find_GSL
def test_assertExpressionsEqual(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.e1 = pe.Expression(expr=m.x ** 2 + m.x - 1)
    m.e2 = pe.Expression(expr=m.x ** 2 + m.x - 1)
    m.f = pe.Expression(expr=m.x ** 2 + 2 * m.x - 1)
    m.g = pe.Expression(expr=m.x ** 2 + m.x - 2)
    assertExpressionsEqual(self, m.e1.expr, m.e2.expr)
    assertExpressionsStructurallyEqual(self, m.e1.expr, m.e2.expr)
    with self.assertRaisesRegex(AssertionError, 'Expressions not equal:'):
        assertExpressionsEqual(self, m.e1.expr, m.f.expr)
    with self.assertRaisesRegex(AssertionError, 'Expressions not structurally equal:'):
        assertExpressionsStructurallyEqual(self, m.e1.expr, m.f.expr)
    i = m.clone()
    with self.assertRaisesRegex(AssertionError, 'Expressions not equal:'):
        assertExpressionsEqual(self, m.e1.expr, i.e1.expr)
    assertExpressionsStructurallyEqual(self, m.e1.expr, i.e1.expr)