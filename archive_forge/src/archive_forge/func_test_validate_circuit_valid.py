import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
def test_validate_circuit_valid():
    qubits = cirq.LineQubit.range(10)
    device = ionq.IonQAPIDevice(qubits)
    for _ in range(100):
        circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=3, op_density=0.5, gate_domain={gate: gate.num_qubits() for gate in VALID_GATES})
        device.validate_circuit(circuit)