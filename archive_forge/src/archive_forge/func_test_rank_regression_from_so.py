from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixReductions
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x
def test_rank_regression_from_so():
    nu, lamb = symbols('nu, lambda')
    A = Matrix([[-3 * nu, 1, 0, 0], [3 * nu, -2 * nu - 1, 2, 0], [0, 2 * nu, -1 * nu - lamb - 2, 3], [0, 0, nu + lamb, -3]])
    expected_reduced = Matrix([[1, 0, 0, 1 / (nu ** 2 * (-lamb - nu))], [0, 1, 0, 3 / (nu * (-lamb - nu))], [0, 0, 1, 3 / (-lamb - nu)], [0, 0, 0, 0]])
    expected_pivots = (0, 1, 2)
    reduced, pivots = A.rref()
    assert simplify(expected_reduced - reduced) == zeros(*A.shape)
    assert pivots == expected_pivots