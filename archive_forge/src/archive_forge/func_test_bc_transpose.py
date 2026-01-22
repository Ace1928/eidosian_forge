from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_bc_transpose():
    assert bc_transpose(Transpose(BlockMatrix([[A, B], [C, D]]))) == BlockMatrix([[A.T, C.T], [B.T, D.T]])