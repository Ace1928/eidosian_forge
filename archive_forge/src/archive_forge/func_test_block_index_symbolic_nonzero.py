from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def test_block_index_symbolic_nonzero():
    k, l, m, n = symbols('k l m n', integer=True, positive=True)
    i, j = symbols('i j', integer=True, nonnegative=True)
    A1 = MatrixSymbol('A1', n, k)
    A2 = MatrixSymbol('A2', n, l)
    A3 = MatrixSymbol('A3', m, k)
    A4 = MatrixSymbol('A4', m, l)
    A = BlockMatrix([[A1, A2], [A3, A4]])
    assert A[0, 0] == A1[0, 0]
    assert A[n + m - 1, 0] == A3[m - 1, 0]
    assert A[0, k + l - 1] == A2[0, l - 1]
    assert A[n + m - 1, k + l - 1] == A4[m - 1, l - 1]
    assert A[i, j] == MatrixElement(A, i, j)
    assert A[n + i, k + j] == A4[i, j]
    assert A[n - i - 1, k - j - 1] == A1[n - i - 1, k - j - 1]
    assert A[2 * n, 2 * k] == A4[n, k]