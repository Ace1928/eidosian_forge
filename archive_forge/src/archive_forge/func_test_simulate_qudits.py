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
def test_simulate_qudits(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    circuit = cirq.Circuit(cirq.H(q0), cirq.XPowGate(dimension=3)(q1) ** 2)
    result = simulator.simulate(circuit, qubit_order=[q1, q0])
    expected = np.zeros((6, 6))
    expected[4:, 4:] = np.ones((2, 2)) / 2
    np.testing.assert_almost_equal(result.final_density_matrix, expected)
    assert len(result.measurements) == 0