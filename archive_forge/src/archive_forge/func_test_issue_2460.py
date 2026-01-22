from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_issue_2460():
    bdm1 = BlockDiagMatrix(Matrix([i]), Matrix([j]))
    bdm2 = BlockDiagMatrix(Matrix([k]), Matrix([l]))
    assert block_collapse(bdm1 + bdm2) == BlockDiagMatrix(Matrix([i + k]), Matrix([j + l]))