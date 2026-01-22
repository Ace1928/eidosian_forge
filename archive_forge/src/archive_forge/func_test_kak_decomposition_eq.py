import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_decomposition_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.KakDecomposition(global_phase=1, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.Y)), interaction_coefficients=(0.3, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z))))
    eq.add_equality_group(cirq.KakDecomposition(global_phase=-1, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.Y)), interaction_coefficients=(0.3, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z))))
    eq.add_equality_group(cirq.KakDecomposition(global_phase=1, single_qubit_operations_before=(np.eye(2), np.eye(2)), interaction_coefficients=(0.3, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), np.eye(2))), cirq.KakDecomposition(interaction_coefficients=(0.3, 0.2, 0.1)))
    eq.make_equality_group(lambda: cirq.KakDecomposition(global_phase=1, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.H)), interaction_coefficients=(0.3, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z))))
    eq.make_equality_group(lambda: cirq.KakDecomposition(global_phase=1, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.Y)), interaction_coefficients=(0.5, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z))))