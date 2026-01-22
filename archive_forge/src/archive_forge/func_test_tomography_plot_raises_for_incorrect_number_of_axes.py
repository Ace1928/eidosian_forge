import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
@pytest.mark.usefixtures('closefigures')
def test_tomography_plot_raises_for_incorrect_number_of_axes():
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    circuit = circuits.Circuit(ops.X(qubit) ** 0.5)
    result = single_qubit_state_tomography(simulator, qubit, circuit, 1000)
    with pytest.raises(TypeError):
        ax = plt.subplot()
        result.plot(ax)
    with pytest.raises(ValueError):
        _, axes = plt.subplots(1, 3)
        result.plot(axes)