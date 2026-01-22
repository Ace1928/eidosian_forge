from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_sparse_creation():
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2], [3, 4]]})
    assert a == SparseMatrix([[1, 2], [3, 4]])
    a = SparseMatrix(2, 2, {(0, 0): [[1, 2]]})
    assert a == SparseMatrix([[1, 2], [0, 0]])
    a = SparseMatrix(2, 2, {(0, 0): [1, 2]})
    assert a == SparseMatrix([[1, 0], [2, 0]])