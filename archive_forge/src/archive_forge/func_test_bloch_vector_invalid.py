import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_invalid():
    with pytest.raises(ValueError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5]), 0)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5, 0.5]), -1)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(np.array([0.5, 0.5, 0.5, 0.5]), 2)