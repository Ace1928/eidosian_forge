import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state_product_state():
    q0, q1, q2 = cirq.LineQubit.range(3)
    product_state_1 = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ONE(q2)
    state = cirq.quantum_state(product_state_1)
    np.testing.assert_allclose(state.data, product_state_1.state_vector())
    assert state.qid_shape == (2, 2, 2)
    assert state.dtype == np.complex64
    with pytest.raises(ValueError, match='qid shape'):
        _ = cirq.quantum_state(product_state_1, qid_shape=(2, 2))