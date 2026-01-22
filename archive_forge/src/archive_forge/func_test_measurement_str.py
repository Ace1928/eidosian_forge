import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_measurement_str():
    q0 = cirq.NamedQid('q0', dimension=3)
    circuit = cirq.Circuit(cirq.measure(q0))
    simulator = ccq.mps_simulator.MPSSimulator()
    result = simulator.run(circuit, repetitions=7)
    assert str(result) == 'q0 (d=3)=0000000'