import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_sandwich_operator_identity():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    tot_c = ccq.circuit_for_expectation_value(circuit, cirq.PauliString({}))
    np.testing.assert_allclose(cirq.unitary(tot_c), np.eye(2 ** len(qubits)), atol=1e-06)