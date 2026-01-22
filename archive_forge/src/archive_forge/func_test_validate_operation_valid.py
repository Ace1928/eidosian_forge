import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
@pytest.mark.parametrize('gate', VALID_GATES)
def test_validate_operation_valid(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    device = ionq.IonQAPIDevice(qubits=qubits)
    operation = gate(*qubits)
    device.validate_operation(operation)