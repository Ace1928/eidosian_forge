import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_moment_steps():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q0), cirq.H(q1))
    simulator = cirq.CliffordSimulator()
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step.state.to_numpy(), np.array([0.5] * 4))
        else:
            np.testing.assert_almost_equal(step.state.to_numpy(), np.array([1, 0, 0, 0]))