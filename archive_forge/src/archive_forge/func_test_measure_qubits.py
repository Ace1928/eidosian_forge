import numpy as np
import pytest
import cirq
def test_measure_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure()
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure([])
    assert cirq.measure(a) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure([a]) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    assert cirq.measure([a, b]) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    qubit_generator = (q for q in (a, b))
    assert cirq.measure(qubit_generator) == cirq.MeasurementGate(num_qubits=2, key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(num_qubits=2, key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(num_qubits=1, key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(num_qubits=1, key='a', invert_mask=(True,)).on(a)
    assert cirq.measure(*cirq.LineQid.for_qid_shape((1, 2, 3)), key='a') == cirq.MeasurementGate(num_qubits=3, key='a', qid_shape=(1, 2, 3)).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))
    assert cirq.measure(cirq.LineQid.for_qid_shape((1, 2, 3)), key='a') == cirq.MeasurementGate(num_qubits=3, key='a', qid_shape=(1, 2, 3)).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))
    cmap = {(0,): np.array([[0, 1], [1, 0]])}
    assert cirq.measure(a, confusion_map=cmap) == cirq.MeasurementGate(num_qubits=1, key='a', confusion_map=cmap).on(a)
    with pytest.raises(ValueError, match='ndarray'):
        _ = cirq.measure(np.array([1, 0]))
    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure('bork')
    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure([a, [b]])
    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure([a], [b])