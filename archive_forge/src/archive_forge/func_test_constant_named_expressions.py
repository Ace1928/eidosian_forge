import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_constant_named_expressions(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=3)
    m.e = pyo.Expression(expr=2)
    e = m.x * m.e
    derivs = reverse_ad(e)
    symbolic = reverse_sd(e)
    self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
    self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)