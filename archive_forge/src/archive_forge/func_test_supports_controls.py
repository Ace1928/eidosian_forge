import pytest
import numpy as np
import cirq
def test_supports_controls():
    u = np.array([[2, 3], [5, 7]])
    assert np.allclose(cirq.kron_with_controls(cirq.CONTROL_TAG), np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.kron_with_controls(cirq.CONTROL_TAG, u), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 3], [0, 0, 5, 7]]))
    assert np.allclose(cirq.kron_with_controls(u, cirq.CONTROL_TAG), np.array([[1, 0, 0, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 5, 0, 7]]))