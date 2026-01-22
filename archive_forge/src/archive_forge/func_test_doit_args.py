from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.common import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises
def test_doit_args():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatAdd(A, MatPow(B, 2)).doit() == A + B ** 2
    assert MatAdd(A, MatMul(A, B)).doit() == A + A * B
    assert MatAdd(A, X, MatMul(A, B), Y, MatAdd(2 * A, B)).doit() == add(A, X, MatMul(A, B), Y, add(2 * A, B)).doit() == MatAdd(3 * A + A * B + B, X, Y)