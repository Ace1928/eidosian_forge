import pytest
import cirq
def test_with_observable():
    o1 = [cirq.Z, cirq.Y, cirq.X]
    o2 = [cirq.X, cirq.Y, cirq.Z]
    g1 = cirq.PauliMeasurementGate(o1, key='a')
    g2 = cirq.PauliMeasurementGate(o2, key='a')
    assert g1.with_observable(o2) == g2
    assert g1.with_observable(o1) is g1