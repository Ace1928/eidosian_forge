import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_random_seed_does_not_modify_global_state_terminal_measurements():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a) ** 0.5, cirq.measure(a))
    sim = cirq.DensityMatrixSimulator(seed=1234)
    result1 = sim.run(circuit, repetitions=50)
    sim = cirq.DensityMatrixSimulator(seed=1234)
    _ = np.random.random()
    _ = random.random()
    result2 = sim.run(circuit, repetitions=50)
    assert result1 == result2