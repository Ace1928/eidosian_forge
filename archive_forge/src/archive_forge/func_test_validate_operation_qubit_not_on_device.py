import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
def test_validate_operation_qubit_not_on_device():
    device = ionq.IonQAPIDevice(qubits=[cirq.LineQubit(0)])
    with pytest.raises(ValueError, match='not on the device'):
        device.validate_operation(cirq.H(cirq.LineQubit(1)))