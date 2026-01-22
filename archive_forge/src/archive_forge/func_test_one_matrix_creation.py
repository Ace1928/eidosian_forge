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
def test_one_matrix_creation():
    assert OneMatrix(2, 2)
    assert OneMatrix(0, 0)
    assert Eq(OneMatrix(1, 1), Identity(1))
    raises(ValueError, lambda: OneMatrix(-1, 2))
    raises(ValueError, lambda: OneMatrix(2.0, 2))
    raises(ValueError, lambda: OneMatrix(2j, 2))
    raises(ValueError, lambda: OneMatrix(2, -1))
    raises(ValueError, lambda: OneMatrix(2, 2.0))
    raises(ValueError, lambda: OneMatrix(2, 2j))
    n = symbols('n')
    assert OneMatrix(n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: OneMatrix(n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: OneMatrix(n, n))