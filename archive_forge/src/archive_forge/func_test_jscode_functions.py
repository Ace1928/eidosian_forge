from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_functions():
    assert jscode(sin(x) ** cos(x)) == 'Math.pow(Math.sin(x), Math.cos(x))'
    assert jscode(sinh(x) * cosh(x)) == 'Math.sinh(x)*Math.cosh(x)'
    assert jscode(Max(x, y) + Min(x, y)) == 'Math.max(x, y) + Math.min(x, y)'
    assert jscode(tanh(x) * acosh(y)) == 'Math.tanh(x)*Math.acosh(y)'
    assert jscode(asin(x) - acos(y)) == '-Math.acos(y) + Math.asin(x)'