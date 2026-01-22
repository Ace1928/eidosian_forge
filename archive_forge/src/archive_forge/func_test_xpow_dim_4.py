import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_xpow_dim_4():
    x = cirq.XPowGate(dimension=4)
    assert cirq.X != x
    expected = [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    assert np.allclose(cirq.unitary(x), expected)
    sim = cirq.Simulator()
    circuit = cirq.Circuit([x(cirq.LineQid(0, 4)) ** 0.5] * 8)
    svs = [step.state_vector(copy=True) for step in sim.simulate_moment_steps(circuit)]
    expected = [[0.65, 0.65, 0.27, 0.27], [0.0, 1.0, 0.0, 0.0], [0.27, 0.65, 0.65, 0.27], [0.0, 0.0, 1.0, 0.0], [0.27, 0.27, 0.65, 0.65], [0.0, 0.0, 0.0, 1.0], [0.65, 0.27, 0.27, 0.65], [1.0, 0.0, 0.0, 0.0]]
    assert np.allclose(np.abs(svs), expected, atol=0.01)