import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_bad_wrt(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.23)
    with self.assertRaisesRegex(ValueError, 'Cannot specify both wrt and wrt_list'):
        ddx = differentiate(m.x ** 2, wrt=m.x, wrt_list=[m.x])