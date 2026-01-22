import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_of_state_vector_as_mixture_invalid_input():
    with pytest.raises(ValueError, match='7'):
        cirq.partial_trace_of_state_vector_as_mixture(np.arange(7), [1, 2], atol=1e-08)
    with pytest.raises(ValueError, match='normalized'):
        cirq.partial_trace_of_state_vector_as_mixture(np.arange(8), [1], atol=1e-08)
    state = np.arange(8) / np.linalg.norm(np.arange(8))
    with pytest.raises(ValueError, match='repeated axis'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [1, 2, 2], atol=1e-08)
    state = np.array([1, 0, 0, 0]).reshape((2, 2))
    with pytest.raises(IndexError, match='out of range'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [5], atol=1e-08)
    with pytest.raises(IndexError, match='out of range'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 2], atol=1e-08)