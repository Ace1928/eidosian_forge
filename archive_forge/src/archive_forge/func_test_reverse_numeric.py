import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_reverse_numeric(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.23)
    m.y = pyo.Var(initialize=0.88)
    ddx = differentiate(m.x ** 2, wrt=m.x, mode='reverse_numeric')
    self.assertIsInstance(ddx, float)
    self.assertAlmostEqual(ddx, 0.46)
    ddy = differentiate(m.x ** 2, wrt=m.y, mode='reverse_numeric')
    self.assertEqual(ddy, 0)
    ddx = differentiate(m.x ** 2, wrt_list=[m.x, m.y], mode='reverse_numeric')
    self.assertIsInstance(ddx, list)
    self.assertEqual(len(ddx), 2)
    self.assertIsInstance(ddx[0], float)
    self.assertAlmostEqual(ddx[0], 0.46)
    self.assertEqual(ddx[1], 0)