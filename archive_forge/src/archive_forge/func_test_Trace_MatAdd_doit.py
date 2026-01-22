from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i
def test_Trace_MatAdd_doit():
    X = ImmutableMatrix([[1, 2, 3]] * 3)
    Y = MatrixSymbol('Y', 3, 3)
    q = MatAdd(X, 2 * X, Y, -3 * Y)
    assert Trace(q).arg == q
    assert Trace(q).doit() == 18 - 2 * Trace(Y)