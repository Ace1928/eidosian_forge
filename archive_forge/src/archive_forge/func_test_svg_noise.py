import IPython.display
import numpy as np
import pytest
import cirq
from cirq.contrib.svg import circuit_to_svg
def test_svg_noise():
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=0.001))
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, [q]))
    svg = circuit_to_svg(circuit)
    assert '>D(0.001)</text>' in svg