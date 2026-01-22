import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_probs_dont_sum_up_to_one():
    q0 = cirq.NamedQid('q0', dimension=2)
    circuit = cirq.Circuit(cirq.measure(q0))
    simulator = ccq.mps_simulator.MPSSimulator(simulation_options=ccq.mps_simulator.MPSOptions(sum_prob_atol=-0.5))
    with pytest.raises(ValueError, match='Sum of probabilities exceeds tolerance'):
        simulator.run(circuit, repetitions=1)