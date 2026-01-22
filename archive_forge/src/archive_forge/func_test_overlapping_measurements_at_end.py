import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_overlapping_measurements_at_end():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.measure(a, key='a'), cirq.measure(a, key='not a', invert_mask=(True,)), cirq.measure(b, key='b'), cirq.measure(a, b, key='ab'))
    samples = cirq.Simulator().sample(circuit, repetitions=100)
    np.testing.assert_array_equal(samples['a'].values, samples['not a'].values ^ 1)
    np.testing.assert_array_equal(samples['a'].values * 2 + samples['b'].values, samples['ab'].values)
    counts = samples['b'].value_counts()
    assert len(counts) == 2
    assert 10 <= counts[0] <= 90
    assert 10 <= counts[1] <= 90