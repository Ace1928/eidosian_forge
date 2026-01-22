import numpy as np
import cirq
import pytest
def test_state_prep_channel_kraus_small():
    gate = cirq.StatePreparationChannel(np.array([0.0, 1.0]))(cirq.LineQubit(0))
    np.testing.assert_almost_equal(cirq.kraus(gate), (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])))
    assert cirq.has_kraus(gate)
    assert not cirq.has_mixture(gate)
    gate = cirq.StatePreparationChannel(np.array([1.0, 0.0]))(cirq.LineQubit(0))
    np.testing.assert_almost_equal(cirq.kraus(gate), (np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]])))
    assert cirq.has_kraus(gate)
    assert not cirq.has_mixture(gate)