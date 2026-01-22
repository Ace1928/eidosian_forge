from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_number_uses__eq__():
    assert cirq.approx_eq(C(0), C(0), atol=0.0)
    assert not cirq.approx_eq(X(0), X(1), atol=0.0)
    assert not cirq.approx_eq(X(0), 0, atol=0.0)
    assert not cirq.approx_eq(Y(), 1, atol=0.0)