import numpy as np
import pytest
import cirq
import cirq.testing
@pytest.mark.parametrize('global_phase', (1, 1j, np.exp(1j)))
def test_bloch_vector_plus_state(global_phase):
    sqrt = np.sqrt(0.5)
    plus_state = global_phase * np.array([sqrt, sqrt])
    bloch = cirq.bloch_vector_from_state_vector(plus_state, 0)
    desired_simple = np.array([1, 0, 0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)