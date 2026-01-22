import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_density_matrix():
    assert_valid_density_matrix(np.array([[1, 0], [0, 0]]))
    assert_valid_density_matrix(np.array([[0.5, 0], [0, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.2], [0.2, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]]))
    assert_valid_density_matrix(np.array([[0.5, 0.2 - 0.2j], [0.2 + 0.2j, 0.5]]))
    assert_valid_density_matrix(np.eye(4) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([1, 0, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(np.ones([4, 4]) / 4.0, num_qubits=2)
    assert_valid_density_matrix(np.diag([0.2, 0.8, 0, 0]), num_qubits=2)
    assert_valid_density_matrix(np.array([[0.2, 0, 0, 0.2 - 0.3j], [0, 0, 0, 0], [0, 0, 0, 0], [0.2 + 0.3j, 0, 0, 0.8]]), num_qubits=2)
    assert_valid_density_matrix(np.array([[1, 0, 0]] + [[0, 0, 0]] * 2), qid_shape=(3,))
    assert_valid_density_matrix(np.array([[0, 0, 0], [0, 0.5, 0.5j], [0, -0.5j, 0.5]]), qid_shape=(3,))
    assert_valid_density_matrix(np.eye(9) / 9.0, qid_shape=(3, 3))
    assert_valid_density_matrix(np.eye(12) / 12.0, qid_shape=(3, 4))
    assert_valid_density_matrix(np.ones([9, 9]) / 9.0, qid_shape=(3, 3))
    assert_valid_density_matrix(np.diag([0.2, 0.8, 0, 0]), qid_shape=(4,))