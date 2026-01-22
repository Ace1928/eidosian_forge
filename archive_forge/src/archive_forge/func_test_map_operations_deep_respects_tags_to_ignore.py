from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_deep_respects_tags_to_ignore():
    q = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(cirq.CX(*q), cirq.CX(*q).with_tags('ignore'), cirq.CX(*q))
    c_nested_mapped = cirq.FrozenCircuit(cirq.CZ(*q), cirq.CX(*q).with_tags('ignore'), cirq.CZ(*q))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested, cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CircuitOperation(c_nested).repeat(5).with_tags('preserve_tag'), cirq.CircuitOperation(c_nested).repeat(6).with_tags('ignore'), cirq.CircuitOperation(c_nested).repeat(7))), c_nested)
    c_expected = cirq.Circuit(c_nested_mapped, cirq.CircuitOperation(c_nested).repeat(4).with_tags('ignore'), c_nested_mapped, cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CircuitOperation(c_nested_mapped).repeat(5).with_tags('preserve_tag'), cirq.CircuitOperation(c_nested).repeat(6).with_tags('ignore'), cirq.CircuitOperation(c_nested_mapped).repeat(7))), c_nested_mapped)
    cirq.testing.assert_same_circuits(cirq.map_operations(c_orig, lambda op, _: cirq.CZ(*op.qubits) if op.gate == cirq.CX else op, tags_to_ignore=['ignore'], deep=True), c_expected)