from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_Piecewise_deep():
    p = jscode(2 * Piecewise((x, x < 1), (x ** 2, True)))
    s = '2*((x < 1) ? (\n   x\n)\n: (\n   Math.pow(x, 2)\n))'
    assert p == s