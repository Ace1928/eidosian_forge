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
def test_run_qudit_mixture(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 2))
    mixture = _TestMixture([cirq.XPowGate(dimension=3) ** 0, cirq.XPowGate(dimension=3), cirq.XPowGate(dimension=3) ** 2])
    circuit = cirq.Circuit(mixture(q0), cirq.measure(q0), cirq.measure(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['q(1) (d=2)'], [[0]] * 100)
    q0_measurements = set((x[0] for x in result.measurements['q(0) (d=3)'].tolist()))
    assert q0_measurements == {0, 1, 2}