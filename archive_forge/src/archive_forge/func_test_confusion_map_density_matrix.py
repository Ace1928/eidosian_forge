import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_map_density_matrix():
    q0, q1 = cirq.LineQubit.range(2)
    p_q0 = 0.3
    confusion = np.array([[0.8, 0.2], [0.1, 0.9]])
    circuit = cirq.Circuit(cirq.X(q0) ** (np.arcsin(np.sqrt(p_q0)) * 2 / np.pi), cirq.measure(q0, key='a', confusion_map={(0,): confusion}), cirq.X(q1).with_classical_controls('a'))
    deferred = cirq.defer_measurements(circuit)
    q_order = (q0, q1, _MeasurementQid('a', q0))
    rho = cirq.final_density_matrix(deferred, qubit_order=q_order).reshape((2,) * 6)
    q0_probs = [1 - p_q0, p_q0]
    assert np.allclose(cirq.partial_trace(rho, [0]), np.diag(q0_probs))
    expected = np.diag(q0_probs @ confusion)
    assert np.allclose(cirq.partial_trace(rho, [1]), expected)
    assert np.allclose(cirq.partial_trace(rho, [2]), expected)