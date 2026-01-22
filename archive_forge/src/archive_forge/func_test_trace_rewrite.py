from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i
def test_trace_rewrite():
    assert trace(A).rewrite(Sum) == Sum(A[i, i], (i, 0, n - 1))
    assert trace(eye(3)).rewrite(Sum) == 3