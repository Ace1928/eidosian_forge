from typing import List
import numpy as np
import pytest
import cirq
def test_merge_k_qubit_unitaries_deep():
    q = cirq.LineQubit.range(2)
    h_cz_y = [cirq.H(q[0]), cirq.CZ(*q), cirq.Y(q[1])]
    c_orig = cirq.Circuit(h_cz_y, cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.Y(q[1])), cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags('ignore'), [cirq.CNOT(*q), cirq.CNOT(*q)], cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(4), [cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)], cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(5).with_tags('preserve_tag'))

    def _wrap_in_cop(ops: cirq.OP_TREE, tag: str):
        return cirq.CircuitOperation(cirq.FrozenCircuit(ops)).with_tags(tag)
    c_expected = cirq.Circuit(_wrap_in_cop([h_cz_y, cirq.Y(q[1])], '1'), cirq.Moment(cirq.X(q[0]).with_tags('ignore')), cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags('ignore'), _wrap_in_cop([cirq.CNOT(*q), cirq.CNOT(*q)], '2'), cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_cop(h_cz_y, '3'))).repeat(4), _wrap_in_cop([cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)], '4'), cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_cop(h_cz_y, '5'))).repeat(5).with_tags('preserve_tag'), strategy=cirq.InsertStrategy.NEW)
    component_id = 0

    def rewriter_merge_to_circuit_op(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal component_id
        component_id = component_id + 1
        return op.with_tags(f'{component_id}')
    context = cirq.TransformerContext(tags_to_ignore=('ignore',), deep=True)
    c_new = cirq.merge_k_qubit_unitaries(c_orig, k=2, context=context, rewriter=rewriter_merge_to_circuit_op)
    cirq.testing.assert_same_circuits(c_new, c_expected)

    def _wrap_in_matrix_gate(ops: cirq.OP_TREE):
        op = _wrap_in_cop(ops, 'temp')
        return cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits)
    c_expected_matrix = cirq.Circuit(_wrap_in_matrix_gate([h_cz_y, cirq.Y(q[1])]), cirq.Moment(cirq.X(q[0]).with_tags('ignore')), cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags('ignore'), _wrap_in_matrix_gate([cirq.CNOT(*q), cirq.CNOT(*q)]), cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_matrix_gate(h_cz_y))).repeat(4), _wrap_in_matrix_gate([cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)]), cirq.CircuitOperation(cirq.FrozenCircuit(_wrap_in_matrix_gate(h_cz_y))).repeat(5).with_tags('preserve_tag'), strategy=cirq.InsertStrategy.NEW)
    c_new_matrix = cirq.merge_k_qubit_unitaries(c_orig, k=2, context=context)
    cirq.testing.assert_same_circuits(c_new_matrix, c_expected_matrix)