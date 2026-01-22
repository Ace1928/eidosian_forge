from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_reblock_2x2():
    B = BlockMatrix([[MatrixSymbol('A_%d%d' % (i, j), 2, 2) for j in range(3)] for i in range(3)])
    assert B.blocks.shape == (3, 3)
    BB = reblock_2x2(B)
    assert BB.blocks.shape == (2, 2)
    assert B.shape == BB.shape
    assert B.as_explicit() == BB.as_explicit()