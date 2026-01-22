from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_uses__eq__():
    assert cirq.approx_eq(C(0), C(0), atol=0.0)
    assert not cirq.approx_eq(C(1), C(2), atol=0.0)
    assert cirq.approx_eq([C(0)], [C(0)], atol=0.0)
    assert not cirq.approx_eq([C(1)], [C(2)], atol=0.0)
    assert cirq.approx_eq(complex(0, 0), 0, atol=0.0)
    assert cirq.approx_eq(0, complex(0, 0), atol=0.0)