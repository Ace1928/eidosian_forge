from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_tuple():
    assert cirq.approx_eq((1, 1), (1, 1), atol=0.0)
    assert not cirq.approx_eq((1, 1), (1, 1, 1), atol=0.0)
    assert not cirq.approx_eq((1, 1), (1,), atol=0.0)
    assert cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), atol=0.4)
    assert not cirq.approx_eq((1.1, 1.2, 1.3), (1, 1, 1), atol=0.2)