from sympy.core.add import Add
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.core.relational import Eq
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import (
from sympy.matrices.expressions.matmul import MatMul
from sympy.testing.pytest import raises
def test_generic_zero_matrix():
    z = GenericZeroMatrix()
    n = symbols('n', integer=True)
    A = MatrixSymbol('A', n, n)
    assert z == z
    assert z != A
    assert A != z
    assert z.is_ZeroMatrix
    raises(TypeError, lambda: z.shape)
    raises(TypeError, lambda: z.rows)
    raises(TypeError, lambda: z.cols)
    assert MatAdd() == z
    assert MatAdd(z, A) == MatAdd(A)
    hash(z)