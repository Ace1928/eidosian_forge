import numpy as np
import pytest
import cirq
def test_expectation_from_state_vector_basis_states_single_qubits():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0})
    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([1.0, 0.0]), {q0: 0}), 1.0)
    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([0.0, 1.0]), {q0: 0}), 0.0)