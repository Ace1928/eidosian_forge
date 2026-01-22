import cirq
import numpy as np
import pytest
def test_matrix_mixture_from_unitaries():
    q0 = cirq.LineQubit(0)
    mix = [(0.5, np.array([[1, 0], [0, 1]])), (0.5, np.array([[0, 1], [1, 0]]))]
    half_flip = cirq.MixedUnitaryChannel(mix, key='flip')
    assert cirq.measurement_key_name(half_flip) == 'flip'
    circuit = cirq.Circuit(half_flip.on(q0), cirq.measure(q0, key='m'))
    sim = cirq.Simulator(seed=0)
    results = sim.simulate(circuit)
    assert 'flip' in results.measurements
    assert results.measurements['flip'] == results.measurements['m']