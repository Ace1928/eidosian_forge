from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))
    assert julia_code(2 * pw) == '2 * ((x < 1) ? (x) : (x .^ 2))'
    assert julia_code(pw / x) == '((x < 1) ? (x) : (x .^ 2)) ./ x'
    assert julia_code(pw / (x * y)) == '((x < 1) ? (x) : (x .^ 2)) ./ (x .* y)'
    assert julia_code(pw / 3) == '((x < 1) ? (x) : (x .^ 2)) / 3'