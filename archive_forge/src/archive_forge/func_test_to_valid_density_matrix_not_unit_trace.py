import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_not_unit_trace():
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, -0.1]]), num_qubits=1)
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.zeros([2, 2]), num_qubits=1)