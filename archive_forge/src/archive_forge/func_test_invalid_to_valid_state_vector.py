import numpy as np
import pytest
import cirq
import cirq.testing
def test_invalid_to_valid_state_vector():
    with pytest.raises(ValueError, match='Please specify'):
        _ = cirq.to_valid_state_vector(np.array([1]))
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(np.array([1.0, 0.0], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(-1, 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(5, 2)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector('0000', 2)
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector('not an int', 2)
    with pytest.raises(ValueError, match='num_qubits != len\\(qid_shape\\)'):
        _ = cirq.to_valid_state_vector(0, 5, qid_shape=(1, 2, 3))
    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([3], qid_shape=(3,))
    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([-1], qid_shape=(3,))
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector([], qid_shape=(3,))
    with pytest.raises(ValueError, match='Invalid quantum state'):
        _ = cirq.to_valid_state_vector([0, 1], num_qubits=3)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector([1, 0], qid_shape=(2, 1))
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.int64), qid_shape=(2, 1))