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
def test_issue_8240():
    x, y = symbols('x y')
    n = 200
    diagonal_variables = [Symbol('x%s' % i) for i in range(n)]
    M = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        M[i][i] = diagonal_variables[i]
    M = Matrix(M)
    eigenvals = M.eigenvals()
    assert len(eigenvals) == n
    for i in range(n):
        assert eigenvals[diagonal_variables[i]] == 1
    eigenvals = M.eigenvals(multiple=True)
    assert set(eigenvals) == set(diagonal_variables)
    M = Matrix([[x, 0, 0], [1, y, 0], [2, 3, x]])
    eigenvals = M.eigenvals()
    assert eigenvals == {x: 2, y: 1}
    eigenvals = M.eigenvals(multiple=True)
    assert len(eigenvals) == 3
    assert eigenvals.count(x) == 2
    assert eigenvals.count(y) == 1