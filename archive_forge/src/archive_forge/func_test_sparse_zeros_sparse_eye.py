from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_sparse_zeros_sparse_eye():
    assert SparseMatrix.eye(3) == eye(3, cls=SparseMatrix)
    assert len(SparseMatrix.eye(3).todok()) == 3
    assert SparseMatrix.zeros(3) == zeros(3, cls=SparseMatrix)
    assert len(SparseMatrix.zeros(3).todok()) == 0