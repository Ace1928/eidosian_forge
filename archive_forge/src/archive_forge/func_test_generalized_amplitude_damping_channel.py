import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_generalized_amplitude_damping_channel():
    d = cirq.generalized_amplitude_damp(0.1, 0.3)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(0.1) * np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - 0.3)]]), np.sqrt(0.1) * np.array([[0.0, np.sqrt(0.3)], [0.0, 0.0]]), np.sqrt(0.9) * np.array([[np.sqrt(1.0 - 0.3), 0.0], [0.0, 1.0]]), np.sqrt(0.9) * np.array([[0.0, 0.0], [np.sqrt(0.3), 0.0]])))
    cirq.testing.assert_consistent_channel(d)
    assert not cirq.has_mixture(d)