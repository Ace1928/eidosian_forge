import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_single_qubit_gate():
    q = cirq.LineQubit(0)
    mat = cirq.testing.random_unitary(2)
    gate = cirq.MatrixGate(mat, qid_shape=(2,))
    circuit = cirq.Circuit(gate(q))
    compiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    ops = list(compiled_circuit.all_operations())
    assert len(ops) == 1
    assert isinstance(ops[0].gate, cirq.PhasedXZGate)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, compiled_circuit, atol=1e-08)