import numpy as np
import pytest
import cirq
def test_fidelity_fail_inference():
    state_vector = cirq.one_hot(shape=(4,), dtype=np.complex128)
    state_tensor = np.reshape(state_vector, (2, 2))
    with pytest.raises(ValueError, match='Please specify'):
        _ = cirq.fidelity(state_tensor, 4)