import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_multi_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    np.testing.assert_almost_equal(cirq.kraus(d), (np.sqrt(0.8) * np.eye(4), np.sqrt(0.2) * np.kron(X, X)))
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)
    np.testing.assert_equal(d._num_qubits_(), 2)
    with pytest.raises(ValueError, match='num_qubits should be 1'):
        assert d.p_i == 1.0
    with pytest.raises(ValueError, match='num_qubits should be 1'):
        assert d.p_x == 0.0
    with pytest.raises(ValueError, match='num_qubits should be 1'):
        assert d.p_y == 0.0
    with pytest.raises(ValueError, match='num_qubits should be 1'):
        assert d.p_z == 0.0