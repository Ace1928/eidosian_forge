import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_expectation_values(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    psum1 = cirq.Z(q0) + 3.2 * cirq.Z(q1)
    psum2 = -1 * cirq.X(q0) + 2 * cirq.X(q1)
    c1 = cirq.Circuit(cirq.I(q0), cirq.X(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.simulate_expectation_values(c1, [psum1, psum2])
    assert cirq.approx_eq(result[0], -2.2, atol=1e-06)
    assert cirq.approx_eq(result[1], 0, atol=1e-06)
    c2 = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate_expectation_values(c2, [psum1, psum2])
    assert cirq.approx_eq(result[0], 0, atol=1e-06)
    assert cirq.approx_eq(result[1], 1, atol=1e-06)
    psum3 = cirq.Z(q0) + cirq.X(q1)
    c3 = cirq.Circuit(cirq.I(q0), cirq.H(q1))
    result = simulator.simulate_expectation_values(c3, psum3)
    assert cirq.approx_eq(result[0], 2, atol=1e-06)