from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_respects_tags_to_ignore():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(*q), cirq.Moment(cirq.X(q[0]), cirq.Y(q[1]).with_tags('ignore')), cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.Y(q[1])), cirq.CZ(*q), [cirq.CNOT(*q), cirq.CNOT(*q).with_tags('ignore'), cirq.CNOT(*q)], cirq.CZ(*q))
    c_merged = cirq.Circuit(cirq.Moment(cirq.CZ(*q)), cirq.Moment(cirq.Y(q[1]).with_tags('ignore')), cirq.Moment(cirq.X(q[0]).with_tags('ignore')), cirq.Moment(cirq.CZ(*q)), cirq.Moment(), cirq.Moment(cirq.CNOT(*q).with_tags('ignore')), cirq.Moment(cirq.CZ(*q)), cirq.Moment())

    def merge_func(op1, op2):
        """Artificial example where a CZ will absorb any merge-able operation."""
        return op1 if op1.gate == cirq.CZ else op2 if op2.gate == cirq.CZ else None
    cirq.testing.assert_same_circuits(cirq.merge_operations(c, merge_func, tags_to_ignore=['ignore']), c_merged)