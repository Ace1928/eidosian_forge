import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_nondeterministic_mixture_noise():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q), cirq.measure(q))
    simulator = ccq.mps_simulator.MPSSimulator(noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(0.5)))
    result1 = simulator.run(circuit, repetitions=50)
    result2 = simulator.run(circuit, repetitions=50)
    assert result1 != result2