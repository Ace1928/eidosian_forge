import pytest
import cirq
def test_bad_observable_raises():
    with pytest.raises(ValueError, match='Pauli observable .* is empty'):
        _ = cirq.PauliMeasurementGate([])
    with pytest.raises(ValueError, match='Pauli observable .* must be Iterable\\[`cirq.Pauli`\\]'):
        _ = cirq.PauliMeasurementGate([cirq.I, cirq.X, cirq.Y])
    with pytest.raises(ValueError, match='Pauli observable .* must be Iterable\\[`cirq.Pauli`\\]'):
        _ = cirq.PauliMeasurementGate(cirq.DensePauliString('XYZI'))
    with pytest.raises(ValueError, match='must have coefficient \\+1/-1.'):
        _ = cirq.PauliMeasurementGate(cirq.DensePauliString('XYZ', coefficient=1j))