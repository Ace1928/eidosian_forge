import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_separated_measurements():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.H(a), cirq.H(b), cirq.CZ(a, b), cirq.measure(a, key='a'), cirq.CZ(a, b), cirq.H(b), cirq.measure(b, key='zero')])
    sample = cirq.Simulator().sample(c, repetitions=10)
    np.testing.assert_array_equal(sample['zero'].values, [0] * 10)