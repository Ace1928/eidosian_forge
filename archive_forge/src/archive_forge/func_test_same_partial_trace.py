import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_same_partial_trace():
    qubit_order = cirq.LineQubit.range(2)
    q0, q1 = qubit_order
    mps_simulator = ccq.mps_simulator.MPSSimulator()
    for _ in range(50):
        for initial_state in range(4):
            circuit = cirq.testing.random_circuit(qubit_order, 3, 0.9)
            expected_density_matrix = cirq.final_density_matrix(circuit, qubit_order=qubit_order, initial_state=initial_state)
            expected_partial_trace = cirq.partial_trace(expected_density_matrix.reshape(2, 2, 2, 2), keep_indices=[0])
            final_state = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state).final_state
            actual_density_matrix = final_state.partial_trace([q0, q1])
            actual_partial_trace = final_state.partial_trace([q0])
            np.testing.assert_allclose(actual_density_matrix, expected_density_matrix, atol=0.0001)
            np.testing.assert_allclose(actual_partial_trace, expected_partial_trace, atol=0.0001)