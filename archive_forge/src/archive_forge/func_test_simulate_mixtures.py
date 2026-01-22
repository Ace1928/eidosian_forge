import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_simulate_mixtures(dtype: Type[np.complexfloating], split: bool):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split)
    circuit = cirq.Circuit(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
    count = 0
    for _ in range(100):
        result = simulator.simulate(circuit, qubit_order=[q0])
        if result.measurements['q(0)']:
            np.testing.assert_almost_equal(result.final_state_vector, np.array([0, 1]))
            count += 1
        else:
            np.testing.assert_almost_equal(result.final_state_vector, np.array([1, 0]))
    assert count < 80 and count > 20