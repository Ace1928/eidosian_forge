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
def test_rref():
    e = ReductionsOnlyMatrix(0, 0, [])
    assert e.rref(pivots=False) == e
    e = ReductionsOnlyMatrix(1, 1, [1])
    a = ReductionsOnlyMatrix(1, 1, [5])
    assert e.rref(pivots=False) == a.rref(pivots=False) == e
    a = ReductionsOnlyMatrix(3, 1, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1], [0], [0]])
    a = ReductionsOnlyMatrix(1, 3, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1, 2, 3]])
    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert a.rref(pivots=False) == Matrix([[1, 0, -1], [0, 1, 2], [0, 0, 0]])
    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    b = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 0, 0, 0, 0, 0, 0])
    c = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    d = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 0, 0, 0, 1, 2, 3])
    assert a.rref(pivots=False) == b.rref(pivots=False) == c.rref(pivots=False) == d.rref(pivots=False) == b
    e = eye_Reductions(3)
    z = zeros_Reductions(3)
    assert e.rref(pivots=False) == e
    assert z.rref(pivots=False) == z
    a = ReductionsOnlyMatrix([[0, 0, 1, 2, 2, -5, 3], [-1, 5, 2, 2, 1, -7, 5], [0, 0, -2, -3, -3, 8, -5], [-1, 5, 0, -1, -2, 1, 0]])
    mat, pivot_offsets = a.rref()
    assert mat == Matrix([[1, -5, 0, 0, 1, 1, -1], [0, 0, 1, 0, 0, -1, 1], [0, 0, 0, 1, 1, -2, 1], [0, 0, 0, 0, 0, 0, 0]])
    assert pivot_offsets == (0, 2, 3)
    a = ReductionsOnlyMatrix([[Rational(1, 19), Rational(1, 5), 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    assert a.rref(pivots=False) == Matrix([[1, 0, 0, Rational(-76, 157)], [0, 1, 0, Rational(-5, 157)], [0, 0, 1, Rational(238, 157)], [0, 0, 0, 0]])
    x = Symbol('x')
    a = ReductionsOnlyMatrix(2, 3, [x, 1, 1, sqrt(x), x, 1])
    for i, j in zip(a.rref(pivots=False), [1, 0, sqrt(x) * (-x + 1) / (-x ** Rational(5, 2) + x), 0, 1, 1 / (sqrt(x) + x + 1)]):
        assert simplify(i - j).is_zero