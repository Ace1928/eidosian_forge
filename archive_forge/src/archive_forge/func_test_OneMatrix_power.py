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
def test_OneMatrix_power():
    o = OneMatrix(3, 3)
    assert o ** 0 == Identity(3)
    assert o ** 1 == o
    assert o * o == o ** 2 == 3 * o
    assert o * o * o == o ** 3 == 9 * o
    o = OneMatrix(n, n)
    assert o * o == o ** 2 == n * o
    assert powsimp(o ** (n - 1) * o) == o ** n == n ** (n - 1) * o