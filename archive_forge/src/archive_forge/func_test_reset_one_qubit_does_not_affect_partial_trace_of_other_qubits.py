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
def test_reset_one_qubit_does_not_affect_partial_trace_of_other_qubits(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.reset(q0))
    result = simulator.simulate(circuit)
    expected = np.zeros((4, 4), dtype=dtype)
    expected[0, 0] = 0.5
    expected[1, 1] = 0.5
    np.testing.assert_almost_equal(result.final_density_matrix, expected)