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
def test_echelon_form():
    a = zeros_Reductions(3)
    e = eye_Reductions(3)
    assert a.echelon_form() == a
    assert e.echelon_form() == e
    a = ReductionsOnlyMatrix(0, 0, [])
    assert a.echelon_form() == a
    a = ReductionsOnlyMatrix(1, 1, [5])
    assert a.echelon_form() == a

    def verify_row_null_space(mat, rows, nulls):
        for v in nulls:
            assert all((t.is_zero for t in a_echelon * v))
        for v in rows:
            if not all((t.is_zero for t in v)):
                assert not all((t.is_zero for t in a_echelon * v.transpose()))
    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    nulls = [Matrix([[1], [-2], [1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)
    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    nulls = []
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)
    a = ReductionsOnlyMatrix(3, 3, [2, 1, 3, 0, 0, 0, 2, 1, 3])
    nulls = [Matrix([[Rational(-1, 2)], [1], [0]]), Matrix([[Rational(-3, 2)], [0], [1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)
    a = ReductionsOnlyMatrix(3, 3, [2, 1, 3, 0, 0, 0, 1, 1, 3])
    nulls = [Matrix([[0], [-3], [1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)
    a = ReductionsOnlyMatrix(3, 3, [0, 3, 3, 0, 2, 2, 0, 1, 1])
    nulls = [Matrix([[1], [0], [0]]), Matrix([[0], [-1], [1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)
    a = ReductionsOnlyMatrix(2, 3, [2, 2, 3, 3, 3, 0])
    nulls = [Matrix([[-1], [1], [0]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)