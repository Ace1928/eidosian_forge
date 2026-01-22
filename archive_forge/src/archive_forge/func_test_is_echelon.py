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
def test_is_echelon():
    zro = zeros_Reductions(3)
    ident = eye_Reductions(3)
    assert zro.is_echelon
    assert ident.is_echelon
    a = ReductionsOnlyMatrix(0, 0, [])
    assert a.is_echelon
    a = ReductionsOnlyMatrix(2, 3, [3, 2, 1, 0, 0, 6])
    assert a.is_echelon
    a = ReductionsOnlyMatrix(2, 3, [0, 0, 6, 3, 2, 1])
    assert not a.is_echelon
    x = Symbol('x')
    a = ReductionsOnlyMatrix(3, 1, [x, 0, 0])
    assert a.is_echelon
    a = ReductionsOnlyMatrix(3, 1, [x, x, 0])
    assert not a.is_echelon
    a = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    assert not a.is_echelon