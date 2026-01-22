import numpy as np
import pytest
import cirq
def test_initial_state_bad_shape():
    qubits = cirq.LineQubit.range(3)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((4,), 1 / 2), dtype=np.complex64)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((2, 2), 1 / 2), dtype=np.complex64)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((4, 4), 1 / 4), dtype=np.complex64)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        cirq.DensityMatrixSimulationState(qubits=qubits, initial_state=np.full((2, 2, 2, 2), 1 / 4), dtype=np.complex64)