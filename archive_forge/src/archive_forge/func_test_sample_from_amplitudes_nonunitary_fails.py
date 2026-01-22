import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_sample_from_amplitudes_nonunitary_fails():
    q0, q1 = cirq.LineQubit.range(2)
    sim = cirq.Simulator(seed=1)
    circuit1 = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'))
    with pytest.raises(ValueError, match='does not support intermediate measurement'):
        _ = sim.sample_from_amplitudes(circuit1, {}, sim._prng)
    circuit2 = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.amplitude_damp(0.01)(q0), cirq.amplitude_damp(0.01)(q1))
    with pytest.raises(ValueError, match='does not support non-unitary'):
        _ = sim.sample_from_amplitudes(circuit2, {}, sim._prng)