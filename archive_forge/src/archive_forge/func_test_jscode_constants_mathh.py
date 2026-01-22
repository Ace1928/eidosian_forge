from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_constants_mathh():
    assert jscode(exp(1)) == 'Math.E'
    assert jscode(pi) == 'Math.PI'
    assert jscode(oo) == 'Number.POSITIVE_INFINITY'
    assert jscode(-oo) == 'Number.NEGATIVE_INFINITY'