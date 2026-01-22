from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_eval_partial_derivative_single_1st_rank_tensors_by_tensor():
    expr1 = PartialDerivative(A(i), A(j))
    assert expr1._perform_derivative() - L.delta(i, -j) == 0
    expr2 = PartialDerivative(A(i), A(-j))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, j) == 0
    expr3 = PartialDerivative(A(-i), A(-j))
    assert expr3._perform_derivative() - L.delta(-i, j) == 0
    expr4 = PartialDerivative(A(-i), A(j))
    assert expr4._perform_derivative() - L.metric(-i, -L_0) * L.delta(L_0, -j) == 0
    expr5 = PartialDerivative(A(i), B(j))
    expr6 = PartialDerivative(A(i), C(j))
    expr7 = PartialDerivative(A(i), D(j))
    expr8 = PartialDerivative(A(i), H(j, k))
    assert expr5._perform_derivative() == 0
    assert expr6._perform_derivative() == 0
    assert expr7._perform_derivative() == 0
    assert expr8._perform_derivative() == 0
    expr9 = PartialDerivative(A(i), A(i))
    assert expr9._perform_derivative() - L.delta(L_0, -L_0) == 0
    expr10 = PartialDerivative(A(-i), A(-i))
    assert expr10._perform_derivative() - L.delta(-L_0, L_0) == 0