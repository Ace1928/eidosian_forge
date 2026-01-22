import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_circuit_to_unitary_matrix(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).unitary(), np.array([[1j, 1], [1, 1j]]) * np.sqrt(0.5), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.Y(a) ** 0.25).unitary(), cirq.unitary(cirq.Y(a) ** 0.25), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.Z(a), cirq.X(b)).unitary(), np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.Z(a), cirq.X(b), cirq.CNOT(a, b)).unitary(), np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.H(b), cirq.CNOT(b, a) ** 0.5, cirq.Y(a) ** 0.5).unitary(), np.array([[1, 1, -1, -1], [1j, -1j, -1j, 1j], [1, 1, 1, 1], [1, -1, 1, -1]]) * np.sqrt(0.25), atol=1e-08)
    c = circuit_cls(cirq.measure(a))
    with pytest.raises(ValueError):
        _ = c.unitary(ignore_terminal_measurements=False)
    c = circuit_cls(cirq.measure(a))
    cirq.testing.assert_allclose_up_to_global_phase(c.unitary(), np.eye(2), atol=1e-08)
    c = circuit_cls(cirq.Z(a), cirq.measure(a), cirq.Z(b))
    cirq.testing.assert_allclose_up_to_global_phase(c.unitary(), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), atol=1e-08)
    c = circuit_cls(cirq.measure(a))
    with pytest.raises(ValueError, match='measurement'):
        _ = (c.unitary(ignore_terminal_measurements=False),)
    c = circuit_cls(cirq.measure(a), cirq.X(a))
    with pytest.raises(ValueError):
        _ = c.unitary()
    c = circuit_cls(cirq.measure(a), cirq.measure(b), cirq.CNOT(a, b))
    with pytest.raises(ValueError):
        _ = c.unitary()

    class MysteryGate(cirq.testing.TwoQubitGate):
        pass
    c = circuit_cls(MysteryGate()(a, b))
    with pytest.raises(TypeError):
        _ = c.unitary()
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.measure(a, invert_mask=(True,))).unitary(), cirq.unitary(cirq.X), atol=1e-08)
    c = circuit_cls(cirq.X(a))
    assert c.unitary(dtype=np.complex64).dtype == np.complex64
    assert c.unitary(dtype=np.complex128).dtype == np.complex128
    assert c.unitary(dtype=np.float64).dtype == np.float64