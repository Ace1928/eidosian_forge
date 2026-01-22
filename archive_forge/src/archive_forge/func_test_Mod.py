from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_Mod():
    assert jscode(Mod(x, y)) == '((x % y) + y) % y'
    assert jscode(Mod(x, x + y)) == '((x % (x + y)) + (x + y)) % (x + y)'
    p1, p2 = symbols('p1 p2', positive=True)
    assert jscode(Mod(p1, p2)) == 'p1 % p2'
    assert jscode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert jscode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert jscode(-Mod(p1, p2)) == '-(p1 % p2)'
    assert jscode(x * Mod(p1, p2)) == 'x*(p1 % p2)'