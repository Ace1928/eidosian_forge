import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_random_seed_non_terminal_measurements_deterministic():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a) ** 0.5, cirq.measure(a, key='a'), cirq.X(a) ** 0.5, cirq.measure(a, key='b'))
    sim = cirq.DensityMatrixSimulator(seed=1234)
    result = sim.run(circuit, repetitions=30)
    assert np.all(result.measurements['a'] == [[0], [0], [1], [0], [1], [0], [1], [0], [1], [1], [0], [0], [1], [0], [0], [1], [1], [1], [0], [0], [0], [0], [1], [0], [0], [0], [1], [1], [1], [1]])
    assert np.all(result.measurements['b'] == [[1], [1], [0], [1], [1], [1], [1], [1], [0], [1], [1], [0], [1], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [1], [1], [1], [0], [1], [1], [1]])