import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_pad_tableau():
    tableau = cirq.CliffordTableau(num_qubits=1)
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=2, axes=[0])
    assert padded_tableau == cirq.CliffordTableau(num_qubits=2)
    tableau = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=1, axes=[0])
    assert padded_tableau == cirq.CliffordGate.X.clifford_tableau
    tableau = cirq.CliffordGate.H.clifford_tableau
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=2, axes=[0])
    np.testing.assert_equal(padded_tableau.matrix().astype(np.int64), np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    np.testing.assert_equal(padded_tableau.rs.astype(np.int64), np.zeros(4))
    tableau = cirq.CliffordGate.H.clifford_tableau
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=2, axes=[1])
    np.testing.assert_equal(padded_tableau.matrix().astype(np.int64), np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
    np.testing.assert_equal(padded_tableau.rs.astype(np.int64), np.zeros(4))