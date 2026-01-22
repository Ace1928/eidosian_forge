from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_not_supported():
    assert julia_code(S.ComplexInfinity) == '# Not supported in Julia:\n# ComplexInfinity\nzoo'
    f = Function('f')
    assert julia_code(f(x).diff(x)) == '# Not supported in Julia:\n# Derivative\nDerivative(f(x), x)'