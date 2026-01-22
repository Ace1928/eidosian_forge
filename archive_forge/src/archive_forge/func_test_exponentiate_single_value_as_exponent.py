import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_exponentiate_single_value_as_exponent():
    q = cirq.LineQubit(0)
    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.X(q)), cirq.rx(0.25 * math.pi).on(q))
    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.Y(q)), cirq.ry(0.25 * math.pi).on(q))
    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.Z(q)), cirq.rz(0.25 * math.pi).on(q))
    assert cirq.approx_eq(np.exp(-0.15j * math.pi * cirq.X(q)), cirq.rx(0.3 * math.pi).on(q))
    assert cirq.approx_eq(cirq.X(q) ** 0.5, cirq.XPowGate(exponent=0.5).on(q))
    assert cirq.approx_eq(cirq.Y(q) ** 0.5, cirq.YPowGate(exponent=0.5).on(q))
    assert cirq.approx_eq(cirq.Z(q) ** 0.5, cirq.ZPowGate(exponent=0.5).on(q))