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
def test_final_state_vector(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64), np.array([1j, 1]) * np.sqrt(0.5), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=0, ignore_terminal_measurements=False, dtype=np.complex64), np.array([1j, 1]) * np.sqrt(0.5), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=1, ignore_terminal_measurements=False, dtype=np.complex64), np.array([1, 1j]) * np.sqrt(0.5), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=np.array([1j, 1]) * np.sqrt(0.5), ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 1]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=0, ignore_terminal_measurements=False, dtype=np.complex64), np.array([1, 0, 0, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=1, ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 1, 0, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=2, ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 0, 0, 1]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=3, ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 0, 1, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ZERO(a) * cirq.KET_ZERO(b), ignore_terminal_measurements=False, dtype=np.complex64), np.array([1, 0, 0, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ZERO(a) * cirq.KET_ONE(b), ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 1, 0, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ONE(a) * cirq.KET_ZERO(b), ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 0, 0, 1]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=cirq.KET_ONE(a) * cirq.KET_ONE(b), ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 0, 1, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.measure(a)).final_state_vector(ignore_terminal_measurements=True, dtype=np.complex64), np.array([1, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a), cirq.measure(a)).final_state_vector(ignore_terminal_measurements=True, dtype=np.complex64), np.array([0, 1]), atol=1e-08)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.measure(a), cirq.X(a)).final_state_vector(ignore_terminal_measurements=True, dtype=np.complex64), np.array([1, 0]), atol=1e-08)
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.measure(a)).final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64), np.array([1, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[a, b], ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 1, 0, 0]), atol=1e-08)
    cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[b, a], ignore_terminal_measurements=False, dtype=np.complex64), np.array([0, 0, 1, 0]), atol=1e-08)
    dtypes = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):
        dtypes.append(np.complex256)
    for dt in dtypes:
        cirq.testing.assert_allclose_up_to_global_phase(circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=np.array([1j, 1]) * np.sqrt(0.5), ignore_terminal_measurements=False, dtype=dt), np.array([0, 1]), atol=1e-08)