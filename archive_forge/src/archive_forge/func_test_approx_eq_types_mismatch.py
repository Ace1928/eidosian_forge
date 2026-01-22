from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_types_mismatch():
    assert not cirq.approx_eq(0, A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), 0, atol=0.0)
    assert not cirq.approx_eq(B(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), B(0), atol=0.0)
    assert not cirq.approx_eq(C(0), A(0), atol=0.0)
    assert not cirq.approx_eq(A(0), C(0), atol=0.0)
    assert not cirq.approx_eq(0, [0], atol=1.0)
    assert not cirq.approx_eq([0], 0, atol=0.0)