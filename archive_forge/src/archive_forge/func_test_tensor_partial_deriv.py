from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_tensor_partial_deriv():
    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    assert expr == PartialDerivative(A(i), A(j), A(k))
    assert expr.expr == A(i)
    assert expr.variables == (A(j), A(k))
    assert expr.get_indices() == [i, -j, -k]
    assert expr.get_free_indices() == [i, -j, -k]
    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(i))
    assert expr.expr == A(L_0)
    assert expr.variables == (A(j), A(L_0))
    expr1 = PartialDerivative(A(i), A(j))
    assert expr1.expr == A(i)
    assert expr1.variables == (A(j),)
    expr2 = A(i) * PartialDerivative(H(k, -i), A(j))
    assert expr2.get_indices() == [L_0, k, -L_0, -j]
    expr2b = A(i) * PartialDerivative(H(k, -i), A(-j))
    assert expr2b.get_indices() == [L_0, k, -L_0, j]
    expr3 = A(i) * PartialDerivative(B(k) * C(-i) + 3 * H(k, -i), A(j))
    assert expr3.get_indices() == [L_0, k, -L_0, -j]
    expr4 = (A(i) + B(i)) * PartialDerivative(C(j), D(j))
    assert expr4.get_indices() == [i, L_0, -L_0]
    expr4b = (A(i) + B(i)) * PartialDerivative(C(-j), D(-j))
    assert expr4b.get_indices() == [i, -L_0, L_0]
    expr5 = (A(i) + B(i)) * PartialDerivative(C(-i), D(j))
    assert expr5.get_indices() == [L_0, -L_0, -j]