from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_apply_tag_to_inverted_op_set():
    q = cirq.LineQubit.range(2)
    op = cirq.CNOT(*q)
    tag = 'tag_to_flip'
    c_orig = cirq.Circuit(op, op.with_tags(tag), cirq.CircuitOperation(cirq.FrozenCircuit(op)))
    c_toggled = cirq.Circuit(op.with_tags(tag), op, cirq.CircuitOperation(cirq.FrozenCircuit(op.with_tags(tag))))
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_orig, [tag], deep=True), c_toggled)
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_toggled, [tag], deep=True), c_orig)
    c_toggled = cirq.Circuit(op.with_tags(tag), op, cirq.CircuitOperation(cirq.FrozenCircuit(op)).with_tags(tag))
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_orig, [tag], deep=False), c_toggled)
    cirq.testing.assert_same_circuits(cirq.toggle_tags(c_toggled, [tag], deep=False), c_orig)