from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i
def test_Trace_MutableMatrix_plus():
    X = Matrix([[1, 2], [3, 4]])
    assert Trace(X) + Trace(X) == 2 * Trace(X)