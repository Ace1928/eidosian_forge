from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.testing.pytest import raises
def test_dotproduct_symbolic():
    A = MatrixSymbol('A', 3, 1)
    B = MatrixSymbol('B', 3, 1)
    dot = DotProduct(A, B)
    assert dot.is_scalar == True
    assert unchanged(Mul, 2, dot)
    assert dot * A == (A[0, 0] * B[0, 0] + A[1, 0] * B[1, 0] + A[2, 0] * B[2, 0]) * A