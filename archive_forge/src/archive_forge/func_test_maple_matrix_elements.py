from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import besseli
from sympy.printing.maple import maple_code
def test_maple_matrix_elements():
    A = Matrix([[x, 2, x * y]])
    assert maple_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == 'x^2 + x*y + 2'
    AA = MatrixSymbol('AA', 1, 3)
    assert maple_code(AA) == 'AA'
    assert maple_code(AA[0, 0] ** 2 + sin(AA[0, 1]) + AA[0, 2]) == 'sin(AA[1, 2]) + AA[1, 1]^2 + AA[1, 3]'
    assert maple_code(sum(AA)) == 'AA[1, 1] + AA[1, 2] + AA[1, 3]'