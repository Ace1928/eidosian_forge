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
def test_run_reset(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split)
    circuit = cirq.Circuit(cirq.H(q0), cirq.XPowGate(dimension=3)(q1) ** 2, cirq.reset(q0), cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1a'), cirq.reset(q1), cirq.measure(q1, key='m1b'))
    meas = simulator.run(circuit, repetitions=100).measurements
    assert np.array_equal(meas['m0'], np.zeros((100, 1)))
    assert np.array_equal(meas['m1a'], np.full((100, 1), 2))
    assert np.array_equal(meas['m1b'], np.zeros((100, 1)))