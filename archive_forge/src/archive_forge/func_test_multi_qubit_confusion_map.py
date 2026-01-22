import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_multi_qubit_confusion_map():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q0, q1, key='a', confusion_map={(0, 1): np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.2], [0.2, 0.2, 0.5, 0.1], [0.0, 0.0, 1.0, 0.0]])}), cirq.X(q2).with_classical_controls('a'), cirq.measure(q2, key='b'))
    deferred = cirq.defer_measurements(circuit)
    sim = cirq.DensityMatrixSimulator()
    result = sim.sample(deferred, repetitions=10000)
    assert 5600 <= np.sum(result['a']) <= 6400
    assert 2600 <= np.sum(result['b']) <= 3400
    deferred.insert(0, cirq.X.on_each(q0, q1))
    result = sim.sample(deferred, repetitions=100)
    assert np.sum(result['a']) == 200
    assert np.sum(result['b']) == 100