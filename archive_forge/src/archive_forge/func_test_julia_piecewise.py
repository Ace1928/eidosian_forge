from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))
    assert julia_code(expr) == '((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(expr, assign_to='r') == 'r = ((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(expr, assign_to='r', inline=False) == 'if (x < 1)\n    r = x\nelse\n    r = x .^ 2\nend'
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = '((x < 1) ? (x .^ 2) :\n(x < 2) ? (x .^ 3) :\n(x < 3) ? (x .^ 4) : (x .^ 5))'
    assert julia_code(expr) == expected
    assert julia_code(expr, assign_to='r') == 'r = ' + expected
    assert julia_code(expr, assign_to='r', inline=False) == 'if (x < 1)\n    r = x .^ 2\nelseif (x < 2)\n    r = x .^ 3\nelseif (x < 3)\n    r = x .^ 4\nelse\n    r = x .^ 5\nend'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: julia_code(expr))