import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_simulate_initial_qudit_state(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for b0 in [0, 1, 2]:
        for b1 in [0, 1, 2, 3]:
            circuit = cirq.Circuit(cirq.XPowGate(dimension=3)(q0) ** b0, cirq.XPowGate(dimension=4)(q1) ** b1)
            result = simulator.simulate(circuit, initial_state=6)
            expected_density_matrix = np.zeros(shape=(12, 12))
            expected_density_matrix[(b0 + 1) % 3 * 4 + (b1 + 2) % 4, (b0 + 1) % 3 * 4 + (b1 + 2) % 4] = 1.0
            np.testing.assert_allclose(result.final_density_matrix, expected_density_matrix, atol=1e-15)