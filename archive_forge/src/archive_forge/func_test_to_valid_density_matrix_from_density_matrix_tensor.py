import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_density_matrix_tensor():
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(cirq.one_hot(shape=(2, 2, 2, 2, 2, 2), dtype=np.complex64), num_qubits=3), cirq.one_hot(shape=(8, 8), dtype=np.complex64))
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(cirq.one_hot(shape=(2, 3, 4, 2, 3, 4), dtype=np.complex64), qid_shape=(2, 3, 4)), cirq.one_hot(shape=(24, 24), dtype=np.complex64))