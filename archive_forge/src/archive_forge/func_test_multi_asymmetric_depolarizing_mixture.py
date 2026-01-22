import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_multi_asymmetric_depolarizing_mixture():
    d = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    assert_mixtures_equal(cirq.mixture(d), ((0.8, np.eye(4)), (0.2, np.kron(X, X))))
    assert cirq.has_mixture(d)
    np.testing.assert_equal(d._num_qubits_(), 2)