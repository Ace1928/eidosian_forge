from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_expand_partial_derivative_full_linearity():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)
    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)
    p = PartialDerivative(42, D(j))
    assert p and (not p._expand_partial_derivative())
    expr3a = PartialDerivative(nneg * A(i) + pos * B(i), D(j))
    assert expr3a._expand_partial_derivative() == nneg * PartialDerivative(A(i), D(j)) + pos * PartialDerivative(B(i), D(j))
    expr3b = PartialDerivative(nneg * A(i) + neg * B(i), D(j))
    assert expr3b._expand_partial_derivative() == nneg * PartialDerivative(A(i), D(j)) + neg * PartialDerivative(B(i), D(j))
    expr3c = PartialDerivative(neg * A(i) + pos * B(i), D(j))
    assert expr3c._expand_partial_derivative() == neg * PartialDerivative(A(i), D(j)) + pos * PartialDerivative(B(i), D(j))
    expr3d = PartialDerivative(c1 * A(i) + c2 * B(i), D(j))
    assert expr3d._expand_partial_derivative() == c1 * PartialDerivative(A(i), D(j)) + c2 * PartialDerivative(B(i), D(j))
    expr3e = PartialDerivative(c2 * A(i) + c1 * B(i), D(j))
    assert expr3e._expand_partial_derivative() == c2 * PartialDerivative(A(i), D(j)) + c1 * PartialDerivative(B(i), D(j))
    expr3f = PartialDerivative(c2 * A(i) + c3 * B(i), D(j))
    assert expr3f._expand_partial_derivative() == c2 * PartialDerivative(A(i), D(j)) + c3 * PartialDerivative(B(i), D(j))
    expr3g = PartialDerivative(c3 * A(i) + c2 * B(i), D(j))
    assert expr3g._expand_partial_derivative() == c3 * PartialDerivative(A(i), D(j)) + c2 * PartialDerivative(B(i), D(j))
    expr3h = PartialDerivative(c3 * A(i) + c1 * B(i), D(j))
    assert expr3h._expand_partial_derivative() == c3 * PartialDerivative(A(i), D(j)) + c1 * PartialDerivative(B(i), D(j))
    expr3i = PartialDerivative(c1 * A(i) + c3 * B(i), D(j))
    assert expr3i._expand_partial_derivative() == c1 * PartialDerivative(A(i), D(j)) + c3 * PartialDerivative(B(i), D(j))