from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.matrices import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose
def test_float_eigenvals():
    m = Matrix([[1, 0.6, 0.6], [0.6, 0.9, 0.9], [0.9, 0.6, 0.6]])
    evals = [Rational(5, 4) - sqrt(385) / 20, sqrt(385) / 20 + Rational(5, 4), S.Zero]
    n_evals = m.eigenvals(rational=True, multiple=True)
    n_evals = sorted(n_evals)
    s_evals = [x.evalf() for x in evals]
    s_evals = sorted(s_evals)
    for x, y in zip(n_evals, s_evals):
        assert abs(x - y) < 10 ** (-9)