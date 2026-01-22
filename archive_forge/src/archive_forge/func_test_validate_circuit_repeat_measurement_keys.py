from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_validate_circuit_repeat_measurement_keys(device):
    circuit = cirq.Circuit()
    circuit.append([cirq.measure(cirq.LineQubit(0), key='a'), cirq.measure(cirq.LineQubit(1), key='a')])
    with pytest.raises(ValueError, match='Measurement key a repeated'):
        device.validate_circuit(circuit)