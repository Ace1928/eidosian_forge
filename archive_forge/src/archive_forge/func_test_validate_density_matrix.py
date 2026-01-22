import numpy as np
import pytest
import cirq
import cirq.testing
def test_validate_density_matrix():
    cirq.validate_density_matrix(cirq.testing.random_density_matrix(2), qid_shape=(2,))
    with pytest.raises(ValueError, match='dtype'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0]], dtype=np.complex64), qid_shape=(2,), dtype=np.complex128)
    with pytest.raises(ValueError, match='shape'):
        cirq.to_valid_density_matrix(np.array([[1, 0]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[1, 0.1], [0, 0]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='trace 1'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0.1]]), qid_shape=(2,))
    with pytest.raises(ValueError, match='positive semidefinite'):
        cirq.to_valid_density_matrix(np.array([[1.1, 0], [0, -0.1]], dtype=np.complex64), qid_shape=(2,))