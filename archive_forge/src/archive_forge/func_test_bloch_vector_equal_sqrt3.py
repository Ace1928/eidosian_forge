import numpy as np
import pytest
import cirq
import cirq.testing
def test_bloch_vector_equal_sqrt3():
    sqrt3 = 1 / np.sqrt(3)
    test_state = np.array([0.888074, 0.325058 + 0.325058j])
    bloch = cirq.bloch_vector_from_state_vector(test_state, 0)
    desired_simple = np.array([sqrt3, sqrt3, sqrt3])
    np.testing.assert_array_almost_equal(bloch, desired_simple)