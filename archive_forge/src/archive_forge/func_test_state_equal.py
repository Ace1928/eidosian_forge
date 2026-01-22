import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_state_equal():
    q0, q1 = cirq.LineQubit.range(2)
    state0 = ccq.mps_simulator.MPSState(qubits=(q0,), prng=value.parse_random_state(0), simulation_options=ccq.mps_simulator.MPSOptions(cutoff=0.001, sum_prob_atol=0.001))
    state1a = ccq.mps_simulator.MPSState(qubits=(q1,), prng=value.parse_random_state(0), simulation_options=ccq.mps_simulator.MPSOptions(cutoff=0.001, sum_prob_atol=0.001))
    state1b = ccq.mps_simulator.MPSState(qubits=(q1,), prng=value.parse_random_state(0), simulation_options=ccq.mps_simulator.MPSOptions(cutoff=1729.0, sum_prob_atol=0.001))
    assert state0 == state0
    assert state0 != state1a
    assert state1a != state1b