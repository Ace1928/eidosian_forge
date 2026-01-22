from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_matrix_elements():
    A = Matrix([[x, 2, x * y]])
    assert julia_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == 'x .^ 2 + x .* y + 2'
    A = MatrixSymbol('AA', 1, 3)
    assert julia_code(A) == 'AA'
    assert julia_code(A[0, 0] ** 2 + sin(A[0, 1]) + A[0, 2]) == 'sin(AA[1,2]) + AA[1,1] .^ 2 + AA[1,3]'
    assert julia_code(sum(A)) == 'AA[1,1] + AA[1,2] + AA[1,3]'