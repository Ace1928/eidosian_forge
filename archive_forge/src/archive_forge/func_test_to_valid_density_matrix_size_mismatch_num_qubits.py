import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_size_mismatch_num_qubits():
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]]), num_qubits=2)
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.eye(4) / 4.0, num_qubits=1)