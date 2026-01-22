from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_iterables():

    def gen_1_1():
        yield 1
        yield 1
    assert cirq.approx_eq((1, 1), [1, 1], atol=0.0)
    assert cirq.approx_eq((1, 1), gen_1_1(), atol=0.0)
    assert cirq.approx_eq(gen_1_1(), [1, 1], atol=0.0)