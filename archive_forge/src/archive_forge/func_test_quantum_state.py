import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))
    density_matrix_1 = np.outer(state_vector_1, np.conj(state_vector_1))
    state = cirq.QuantumState(state_vector_1)
    assert state.data is state_vector_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state.state_vector(), state_vector_1)
    np.testing.assert_array_equal(state.state_tensor(), state_tensor_1)
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), state_vector_1)
    state = cirq.QuantumState(state_tensor_1, qid_shape=(2, 2))
    assert state.data is state_tensor_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state.state_vector(), state_vector_1)
    np.testing.assert_array_equal(state.state_tensor(), state_tensor_1)
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), state_vector_1)
    state = cirq.QuantumState(density_matrix_1, qid_shape=(2, 2))
    assert state.data is density_matrix_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    assert state.state_vector() is None
    assert state.state_tensor() is None
    np.testing.assert_array_equal(state.density_matrix(), density_matrix_1)
    np.testing.assert_array_equal(state.state_vector_or_density_matrix(), density_matrix_1)