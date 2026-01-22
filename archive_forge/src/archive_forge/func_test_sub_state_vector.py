import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_sub_state_vector():
    a = np.arange(4) / np.linalg.norm(np.arange(4))
    b = (np.arange(8) + 3) / np.linalg.norm(np.arange(8) + 3)
    c = (np.arange(16) + 1) / np.linalg.norm(np.arange(16) + 1)
    state = np.kron(np.kron(a, b), c).reshape((2, 2, 2, 2, 2, 2, 2, 2, 2))
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(a, [0, 1], atol=1e-08), a)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(b, [0, 1, 2], atol=1e-08), b)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(c, [0, 1, 2, 3], atol=1e-08), c)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(state, [0, 1], atol=1e-15), a.reshape((2, 2)))
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(state, [2, 3, 4], atol=1e-15), b.reshape((2, 2, 2)))
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(state, [5, 6, 7, 8], atol=1e-15), c.reshape((2, 2, 2, 2)))
    reshaped_state = state.reshape(-1)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(reshaped_state, [0, 1], atol=1e-15), a)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(reshaped_state, [2, 3, 4], atol=1e-15), b)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(reshaped_state, [5, 6, 7, 8], atol=1e-15), c)
    assert cirq.sub_state_vector(state, [0, 1], default=None, atol=1e-16) is None
    assert cirq.sub_state_vector(state, [2, 3, 4], default=None, atol=1e-16) is None
    assert cirq.sub_state_vector(state, [5, 6, 7, 8], default=None, atol=1e-16) is None
    for q1 in range(9):
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1) is not None