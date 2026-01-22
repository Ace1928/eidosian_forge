import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_state_vector_tensor():
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(density_matrix_rep=np.array(np.full((2, 2), 0.5), dtype=np.complex64), num_qubits=2), 0.25 * np.ones((4, 4)))