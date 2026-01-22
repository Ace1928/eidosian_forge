import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
@pytest.mark.parametrize('input_prob', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_prob_encoding(input_prob):
    q = cirq.NamedQubit('q')
    gate = ccb.BayesianNetworkGate([('q', input_prob)], [])
    circuit = cirq.Circuit(gate.on(q))
    phi = cirq.Simulator().simulate(circuit, qubit_order=[q], initial_state=0).state_vector(copy=True)
    actual_probs = [abs(x) ** 2 for x in phi]
    np.testing.assert_almost_equal(actual_probs[1], input_prob, decimal=4)