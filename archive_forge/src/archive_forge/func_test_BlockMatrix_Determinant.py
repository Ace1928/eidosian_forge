from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockMatrix_Determinant():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    from sympy.assumptions.ask import Q
    from sympy.assumptions.assume import assuming
    with assuming(Q.invertible(A)):
        assert det(X) == det(A) * det(X.schur('A'))
    assert isinstance(det(X), Expr)
    assert det(BlockMatrix([A])) == det(A)
    assert det(BlockMatrix([ZeroMatrix(n, n)])) == 0