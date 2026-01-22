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
def verify_row_null_space(mat, rows, nulls):
    for v in nulls:
        assert all((t.is_zero for t in a_echelon * v))
    for v in rows:
        if not all((t.is_zero for t in v)):
            assert not all((t.is_zero for t in a_echelon * v.transpose()))