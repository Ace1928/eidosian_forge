from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_does_not_merge_ccos_behind_measurements():
    q = cirq.LineQubit.range(2)
    cco_op = cirq.X(q[1]).with_classical_controls('a')

    def merge_func(op1, op2):
        return cirq.I(*op1.qubits) if op1 == cco_op and op2 == cco_op else None
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.measure(q[0], key='a'), cco_op] * 2)
    cirq.testing.assert_same_circuits(cirq.merge_operations(circuit, merge_func), circuit)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.measure(q[0], key='a'), cco_op, cco_op] * 2)
    expected_circuit = cirq.Circuit([cirq.H(q[0]), cirq.measure(q[0], key='a'), cirq.I(q[1])] * 2)
    cirq.testing.assert_same_circuits(cirq.align_left(cirq.merge_operations(circuit, merge_func)), expected_circuit)