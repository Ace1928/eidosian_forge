from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement
def test_entry_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0)[0, 0] == 1
    assert MatPow(X, 0)[0, 1] == 0
    assert MatPow(X, 1)[0, 0] == 1
    assert MatPow(X, 1)[0, 1] == 2
    assert MatPow(X, 2)[0, 0] == 7