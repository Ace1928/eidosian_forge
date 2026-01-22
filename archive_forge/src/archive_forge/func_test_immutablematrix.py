from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_immutablematrix():
    A = ImmutableMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.shape == (3, 3)
    assert A[1, 2] == 6
    assert A[2, 2] == 9
    assert A[1, :] == ImmutableMatrix([[4, 5, 6]])
    assert A[:2, :2] == ImmutableMatrix([[1, 2], [4, 5]])
    with raises(TypeError):
        A[2, 2] = 5
    X = DenseMatrix([[1, 2], [3, 4]])
    assert X.as_immutable() == ImmutableMatrix([[1, 2], [3, 4]])
    assert X.det() == -2
    X = ImmutableMatrix(eye(3))
    assert isinstance(X + A, ImmutableMatrix)
    assert isinstance(X * A, ImmutableMatrix)
    assert isinstance(X * 2, ImmutableMatrix)
    assert isinstance(2 * X, ImmutableMatrix)
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[1], [0]])
    assert type(X.LUsolve(Y)) == ImmutableMatrix
    x = Symbol('x')
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[1, 2], [x, 4]])
    assert Y.subs(x, 3) == X
    assert Y.xreplace(x, 3) == X
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[5], [6]])
    Z = X.row_join(Y)
    assert isinstance(Z, ImmutableMatrix)
    assert Z == ImmutableMatrix([[1, 2, 5], [3, 4, 6]])
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[5, 6]])
    Z = X.col_join(Y)
    assert isinstance(Z, ImmutableMatrix)
    assert Z == ImmutableMatrix([[1, 2], [3, 4], [5, 6]])
    X = ImmutableMatrix([1])
    Y = DenseMatrix([1])
    assert type(X + Y) == ImmutableMatrix
    assert type(Y + X) == ImmutableMatrix
    assert type(X * Y) == ImmutableMatrix
    assert type(Y * X) == ImmutableMatrix