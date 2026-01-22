import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_tps_initial_state():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X ** b0)(q0), (cirq.X ** b1)(q1))
            result = simulator.simulate(circuit, initial_state=cirq.KET_ZERO(q0) * cirq.KET_ONE(q1))
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + 1 - b1, b0 * 2 + 1 - b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix, expected_density_matrix)