from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_validate_operation_existing_qubits(device):
    device.validate_operation(cirq.GateOperation(cirq.XX, (cirq.LineQubit(0), cirq.LineQubit(1))))
    device.validate_operation(cirq.Z(cirq.LineQubit(0)))
    device.validate_operation(cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.25, global_shift=0.1).on(cirq.LineQubit(1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.Z(cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.NamedQubit('q1')))