import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pytest
import cirq
from cirq.devices import GridQubit
from cirq.vis import state_histogram
def test_get_state_histogram():
    simulator = cirq.Simulator()
    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.X(q1)])
    circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
    result = simulator.run(program=circuit, repetitions=5)
    values_to_plot = state_histogram.get_state_histogram(result)
    expected_values = [0.0, 0.0, 0.0, 5.0]
    np.testing.assert_equal(values_to_plot, expected_values)