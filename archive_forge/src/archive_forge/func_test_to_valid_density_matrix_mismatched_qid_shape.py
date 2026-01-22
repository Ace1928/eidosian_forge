import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_mismatched_qid_shape():
    with pytest.raises(ValueError, match='num_qubits != len\\(qid_shape\\)'):
        cirq.to_valid_density_matrix(np.eye(4) / 4, num_qubits=1, qid_shape=(2, 2))
    with pytest.raises(ValueError, match='num_qubits != len\\(qid_shape\\)'):
        cirq.to_valid_density_matrix(np.eye(4) / 4, num_qubits=2, qid_shape=(4,))
    with pytest.raises(ValueError, match='Both were None'):
        cirq.to_valid_density_matrix(np.eye(4) / 4)