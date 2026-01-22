from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_respects_tags_to_ignore():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(*q), cirq.CNOT(*q).with_tags('ignore'), cirq.CNOT(*q))
    cirq.testing.assert_same_circuits(cirq.Circuit(cirq.Z.on_each(*q), cirq.CNOT(*q).with_tags('ignore'), cirq.Z.on_each(*q)), cirq.map_operations(c, lambda op, i: cirq.Z.on_each(*op.qubits), tags_to_ignore=['ignore']))