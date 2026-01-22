import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_state_vector_creates_new_copy():
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    out = cirq.to_valid_state_vector(state, 2)
    assert out is not state