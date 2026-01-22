from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
@pytest.mark.parametrize('ms', [cirq.Duration(millis=1), timedelta(milliseconds=1)])
def test_init_durations(ms, qubits):
    dev = aqt_device.AQTDevice(qubits=qubits, measurement_duration=100 * ms, twoq_gates_duration=200 * ms, oneq_gates_duration=10 * ms)
    assert dev.metadata.twoq_gates_duration == cirq.Duration(millis=200)
    assert dev.metadata.oneq_gates_duration == cirq.Duration(millis=10)
    assert dev.metadata.measurement_duration == cirq.Duration(millis=100)