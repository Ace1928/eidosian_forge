import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_not_positive_semidefinite():
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(np.array([[0.6, 0.5], [0.5, 0.4]], dtype=np.complex64), num_qubits=1)