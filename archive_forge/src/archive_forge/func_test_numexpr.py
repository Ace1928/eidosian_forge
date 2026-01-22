from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter
def test_numexpr():
    from sympy.logic.boolalg import ITE
    expr = ITE(x > 0, True, False, evaluate=False)
    assert NumExprPrinter().doprint(expr) == "numexpr.evaluate('where((x > 0), True, False)', truediv=True)"
    from sympy.codegen.ast import Return, FunctionDefinition, Variable, Assignment
    func_def = FunctionDefinition(None, 'foo', [Variable(x)], [Assignment(y, x), Return(y ** 2)])
    expected = "def foo(x):\n    y = numexpr.evaluate('x', truediv=True)\n    return numexpr.evaluate('y**2', truediv=True)"
    assert NumExprPrinter().doprint(func_def) == expected