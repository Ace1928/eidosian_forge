from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_expand_partial_derivative_sum_rule():
    tau = symbols('tau')
    expr1aa = PartialDerivative(A(i), tau)
    assert expr1aa._expand_partial_derivative() == PartialDerivative(A(i), tau)
    expr1ab = PartialDerivative(A(i) + B(i), tau)
    assert expr1ab._expand_partial_derivative() == PartialDerivative(A(i), tau) + PartialDerivative(B(i), tau)
    expr1ac = PartialDerivative(A(i) + B(i) + C(i), tau)
    assert expr1ac._expand_partial_derivative() == PartialDerivative(A(i), tau) + PartialDerivative(B(i), tau) + PartialDerivative(C(i), tau)
    expr1ba = PartialDerivative(A(i), D(j))
    assert expr1ba._expand_partial_derivative() == PartialDerivative(A(i), D(j))
    expr1bb = PartialDerivative(A(i) + B(i), D(j))
    assert expr1bb._expand_partial_derivative() == PartialDerivative(A(i), D(j)) + PartialDerivative(B(i), D(j))
    expr1bc = PartialDerivative(A(i) + B(i) + C(i), D(j))
    assert expr1bc._expand_partial_derivative() == PartialDerivative(A(i), D(j)) + PartialDerivative(B(i), D(j)) + PartialDerivative(C(i), D(j))
    expr1ca = PartialDerivative(A(i), H(j, k))
    assert expr1ca._expand_partial_derivative() == PartialDerivative(A(i), H(j, k))
    expr1cb = PartialDerivative(A(i) + B(i), H(j, k))
    assert expr1cb._expand_partial_derivative() == PartialDerivative(A(i), H(j, k)) + PartialDerivative(B(i), H(j, k))
    expr1cc = PartialDerivative(A(i) + B(i) + C(i), H(j, k))
    assert expr1cc._expand_partial_derivative() == PartialDerivative(A(i), H(j, k)) + PartialDerivative(B(i), H(j, k)) + PartialDerivative(C(i), H(j, k))
    expr1da = PartialDerivative(A(i), (D(j), H(k, m)))
    assert expr1da._expand_partial_derivative() == PartialDerivative(A(i), (D(j), H(k, m)))
    expr1db = PartialDerivative(A(i) + B(i), (D(j), H(k, m)))
    assert expr1db._expand_partial_derivative() == PartialDerivative(A(i), (D(j), H(k, m))) + PartialDerivative(B(i), (D(j), H(k, m)))
    expr1dc = PartialDerivative(A(i) + B(i) + C(i), (D(j), H(k, m)))
    assert expr1dc._expand_partial_derivative() == PartialDerivative(A(i), (D(j), H(k, m))) + PartialDerivative(B(i), (D(j), H(k, m))) + PartialDerivative(C(i), (D(j), H(k, m)))