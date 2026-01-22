import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_two_qubits():
    d = cirq.depolarize(0.15, n_qubits=2)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(0.85) * np.eye(4), np.sqrt(0.01) * np.kron(np.eye(2), X), np.sqrt(0.01) * np.kron(np.eye(2), Y), np.sqrt(0.01) * np.kron(np.eye(2), Z), np.sqrt(0.01) * np.kron(X, np.eye(2)), np.sqrt(0.01) * np.kron(X, X), np.sqrt(0.01) * np.kron(X, Y), np.sqrt(0.01) * np.kron(X, Z), np.sqrt(0.01) * np.kron(Y, np.eye(2)), np.sqrt(0.01) * np.kron(Y, X), np.sqrt(0.01) * np.kron(Y, Y), np.sqrt(0.01) * np.kron(Y, Z), np.sqrt(0.01) * np.kron(Z, np.eye(2)), np.sqrt(0.01) * np.kron(Z, X), np.sqrt(0.01) * np.kron(Z, Y), np.sqrt(0.01) * np.kron(Z, Z)))
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)
    assert d.num_qubits() == 2
    cirq.testing.assert_has_diagram(cirq.Circuit(d(*cirq.LineQubit.range(2))), '\n0: ───D(0.15)───\n      │\n1: ───#2────────\n        ')