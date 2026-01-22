import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_trig_functions(self):
    m = ConcreteModel()
    m.x = Var()
    e = differentiate(sin(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(cos(m.x)))
    e = differentiate(cos(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(-1.0 * sin(m.x)))
    e = differentiate(tan(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(1.0 + tan(m.x) ** 2.0))
    e = differentiate(sinh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(cosh(m.x)))
    e = differentiate(cosh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(sinh(m.x)))
    e = differentiate(tanh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(1.0 - tanh(m.x) ** 2.0))
    e = differentiate(asin(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s((1.0 + -1.0 * m.x ** 2.0) ** (-0.5)))
    e = differentiate(acos(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(-1.0 * (1.0 + -1.0 * m.x ** 2.0) ** (-0.5)))
    e = differentiate(atan(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s((1.0 + m.x ** 2.0) ** (-1.0)))
    e = differentiate(asinh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s((1.0 + m.x ** 2) ** (-0.5)))
    e = differentiate(acosh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    if s(e) == s((-1.0 + m.x ** 2.0) ** (-0.5)):
        pass
    else:
        self.assertEqual(s(e), s((1.0 + m.x) ** (-0.5) * (-1.0 + m.x) ** (-0.5)))
    e = differentiate(atanh(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s((1.0 + -1.0 * m.x ** 2.0) ** (-1.0)))