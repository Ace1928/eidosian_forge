import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel():
    d = cirq.depolarize(0.3)
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(0.7) * np.eye(2), np.sqrt(0.1) * X, np.sqrt(0.1) * Y, np.sqrt(0.1) * Z))
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)