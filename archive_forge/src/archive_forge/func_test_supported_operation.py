import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
@pytest.mark.parametrize('op', [cirq.CircuitOperation(cirq.FrozenCircuit(matrix_gate(q[0]))), matrix_gate(q[0]), matrix_gate(q[0]).with_tags('test_tags'), matrix_gate(q[0]).controlled_by(q[1]), matrix_gate(q[0]).controlled_by(q[1]).with_tags('test_tags'), matrix_gate(q[0]).with_tags('test_tags').controlled_by(q[1])])
def test_supported_operation(op):
    circuit = cirq.Circuit(op)
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)
    multi_qubit_ops = [e for e in converted_circuit.all_operations() if len(e.qubits) > 1]
    assert all((isinstance(e.gate, cirq_google.SycamoreGate) for e in multi_qubit_ops))