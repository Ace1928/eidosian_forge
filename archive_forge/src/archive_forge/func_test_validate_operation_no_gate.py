import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
def test_validate_operation_no_gate():
    device = ionq.IonQAPIDevice(qubits=[])
    with pytest.raises(ValueError, match='no gates'):
        device.validate_operation(cirq.CircuitOperation(cirq.FrozenCircuit()))