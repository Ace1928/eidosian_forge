import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pauli_string_expectation_from_state_vector_pure_state():
    qubits = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qubits)}
    circuit = cirq.Circuit(cirq.X(qubits[1]), cirq.H(qubits[2]), cirq.X(qubits[3]), cirq.H(qubits[3]))
    wf = circuit.final_state_vector(qubit_order=qubits, ignore_terminal_measurements=False, dtype=np.complex128)
    z0z1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
    z0z2 = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})
    z0z3 = cirq.PauliString({qubits[0]: cirq.Z, qubits[3]: cirq.Z})
    z0x1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.X})
    z1x2 = cirq.PauliString({qubits[1]: cirq.Z, qubits[2]: cirq.X})
    x0z1 = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    x3 = cirq.PauliString({qubits[3]: cirq.X})
    for state in [wf, wf.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), -1, atol=1e-08)
        np.testing.assert_allclose(z0z2.expectation_from_state_vector(state, q_map), 0, atol=1e-08)
        np.testing.assert_allclose(z0z3.expectation_from_state_vector(state, q_map), 0, atol=1e-08)
        np.testing.assert_allclose(z0x1.expectation_from_state_vector(state, q_map), 0, atol=1e-08)
        np.testing.assert_allclose(z1x2.expectation_from_state_vector(state, q_map), -1, atol=1e-08)
        np.testing.assert_allclose(x0z1.expectation_from_state_vector(state, q_map), 0, atol=1e-08)
        np.testing.assert_allclose(x3.expectation_from_state_vector(state, q_map), -1, atol=1e-08)