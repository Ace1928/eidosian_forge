import numpy as np
import pytest
import cirq
import cirq.contrib.bayesian_network as ccb
@pytest.mark.parametrize('p0,p1,p2,expected_probs', [(0.0, 0.0, 0.0, [1, 0, 0, 0, 0, 0, 0, 0]), (0.0, 0.0, 1.0, [0, 1, 0, 0, 0, 0, 0, 0]), (0.0, 1.0, 0.0, [0, 0, 1, 0, 0, 0, 0, 0]), (0.0, 1.0, 1.0, [0, 0, 0, 1, 0, 0, 0, 0]), (1.0, 0.0, 0.0, [0, 0, 0, 0, 1, 0, 0, 0]), (1.0, 0.0, 1.0, [0, 0, 0, 0, 0, 1, 0, 0]), (1.0, 1.0, 0.0, [0, 0, 0, 0, 0, 0, 1, 0]), (1.0, 1.0, 1.0, [0, 0, 0, 0, 0, 0, 0, 1])])
@pytest.mark.parametrize('decompose', [True, False])
def test_initial_probs(p0, p1, p2, expected_probs, decompose):
    q0, q1, q2 = cirq.LineQubit.range(3)
    gate = ccb.BayesianNetworkGate([('q0', p0), ('q1', p1), ('q2', p2)], [])
    if decompose:
        circuit = cirq.Circuit(cirq.decompose(gate.on(q0, q1, q2)))
    else:
        circuit = cirq.Circuit(gate.on(q0, q1, q2))
    result = cirq.Simulator().simulate(circuit, qubit_order=[q0, q1, q2], initial_state=0)
    actual_probs = [abs(x) ** 2 for x in result.state_vector(copy=True)]
    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-06)