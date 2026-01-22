import numpy as np
import pytest
import cirq
def test_initial_state_matrix():
    qubits = cirq.LineQubit.range(3)
    args = cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((8, 8), 1 / 8), dtype=np.complex64)
    assert args.target_tensor.shape == (2, 2, 2, 2, 2, 2)
    args2 = cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((2, 2, 2, 2, 2, 2), 1 / 8), dtype=np.complex64)
    assert args2.target_tensor.shape == (2, 2, 2, 2, 2, 2)