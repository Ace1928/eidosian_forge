from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_copyin():
    s = SparseMatrix(3, 3, {})
    s[1, 0] = 1
    assert s[:, 0] == SparseMatrix(Matrix([0, 1, 0]))
    assert s[3] == 1
    assert s[3:4] == [1]
    s[1, 1] = 42
    assert s[1, 1] == 42
    assert s[1, 1:] == SparseMatrix([[42, 0]])
    s[1, 1:] = Matrix([[5, 6]])
    assert s[1, :] == SparseMatrix([[1, 5, 6]])
    s[1, 1:] = [[42, 43]]
    assert s[1, :] == SparseMatrix([[1, 42, 43]])
    s[0, 0] = 17
    assert s[:, :1] == SparseMatrix([17, 1, 0])
    s[0, 0] = [1, 1, 1]
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = Matrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])
    s[0, 0] = SparseMatrix([1, 1, 1])
    assert s[:, 0] == SparseMatrix([1, 1, 1])