import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state_computational_basis_state():
    state = cirq.quantum_state(7, qid_shape=(3, 4))
    np.testing.assert_allclose(state.data, cirq.one_hot(index=7, shape=(12,), dtype=np.complex64))
    assert state.qid_shape == (3, 4)
    assert state.dtype == np.complex64
    state = cirq.quantum_state((0, 1, 2, 3), qid_shape=(1, 2, 3, 4), dtype=np.complex128)
    np.testing.assert_allclose(state.data, cirq.one_hot(index=(0, 1, 2, 3), shape=(1, 2, 3, 4), dtype=np.complex64))
    assert state.qid_shape == (1, 2, 3, 4)
    assert state.dtype == np.complex128
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state(7)
    with pytest.raises(ValueError, match='out of range'):
        _ = cirq.quantum_state(7, qid_shape=(2, 2))
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state((0, 1, 2, 3))
    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.quantum_state((0, 1, 2, 3), qid_shape=(2, 2, 2, 2))
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.quantum_state((0, 0, 1, 1), qid_shape=(1, 1, 2, 2))