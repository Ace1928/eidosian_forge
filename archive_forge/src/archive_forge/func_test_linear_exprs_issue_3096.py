import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_linear_exprs_issue_3096(self):
    m = pyo.ConcreteModel()
    m.y1 = pyo.Var(initialize=10)
    m.y2 = pyo.Var(initialize=100)
    e = (m.y1 - 0.5) * (m.y1 - 0.5) + (m.y2 - 0.5) * (m.y2 - 0.5)
    derivs = reverse_ad(e)
    self.assertEqual(derivs[m.y1], 19)
    self.assertEqual(derivs[m.y2], 199)
    symbolic = reverse_sd(e)
    self.assertExpressionsEqual(symbolic[m.y1], m.y1 - 0.5 + m.y1 - 0.5)
    self.assertExpressionsEqual(symbolic[m.y2], m.y2 - 0.5 + m.y2 - 0.5)