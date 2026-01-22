from typing import Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('circuit, qubits', ((cirq.Circuit(cirq.X(Q0) ** 0.25), (Q0,)), (cirq.Circuit(cirq.CNOT(Q0, Q1) ** 0.25), (Q0, Q1)), (cirq.Circuit(cirq.CNOT(Q0, Q1) ** 0.25), (Q1, Q0)), (cirq.Circuit(cirq.TOFFOLI(Q0, Q1, Q2)), (Q1, Q0, Q2)), (cirq.Circuit(cirq.H(Q0), cirq.H(Q1), cirq.CNOT(Q0, Q2), cirq.CNOT(Q1, Q3), cirq.X(Q0), cirq.X(Q1), cirq.CNOT(Q1, Q0)), (Q1, Q0, Q2, Q3))))
def test_density_matrix_from_state_tomography_is_correct(circuit, qubits):
    sim = cirq.Simulator(seed=87539319)
    tomography_result = cirq.experiments.state_tomography(sim, qubits, circuit, repetitions=5000)
    actual_rho = tomography_result.data
    expected_rho = compute_density_matrix(circuit, qubits)
    error_rho = actual_rho - expected_rho
    assert np.linalg.norm(error_rho) < 0.05
    assert np.max(np.abs(error_rho)) < 0.05