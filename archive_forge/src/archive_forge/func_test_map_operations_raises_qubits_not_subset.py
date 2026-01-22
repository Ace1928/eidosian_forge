from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_raises_qubits_not_subset():
    q = cirq.LineQubit.range(3)
    with pytest.raises(ValueError, match='should act on a subset'):
        _ = cirq.map_operations(cirq.Circuit(cirq.CNOT(q[0], q[1])), lambda op, i: cirq.CNOT(q[1], q[2]))