import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state_state_vector_state_tensor():
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))
    state = cirq.quantum_state(state_vector_1, dtype=np.complex64)
    np.testing.assert_array_equal(state.data, state_vector_1)
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex64
    state = cirq.quantum_state(state_tensor_1, qid_shape=(2, 2))
    assert state.data is state_tensor_1
    assert state.qid_shape == (2, 2)
    assert state.dtype == np.complex128
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state(state_tensor_1)
    with pytest.raises(ValueError, match='not compatible'):
        _ = cirq.quantum_state(state_tensor_1, qid_shape=(2, 3))