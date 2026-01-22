import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_damping_channel():
    d = cirq.phase_damp(0.3)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.array([[1.0, 0.0], [0.0, np.sqrt(1 - 0.3)]]), np.array([[0.0, 0.0], [0.0, np.sqrt(0.3)]])))
    cirq.testing.assert_consistent_channel(d)
    assert not cirq.has_mixture(d)