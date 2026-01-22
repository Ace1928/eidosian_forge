import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(0.4) * np.eye(2), np.sqrt(0.1) * X, np.sqrt(0.2) * Y, np.sqrt(0.3) * Z))
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)
    assert cirq.AsymmetricDepolarizingChannel(p_x=0, p_y=0.1, p_z=0).num_qubits() == 1