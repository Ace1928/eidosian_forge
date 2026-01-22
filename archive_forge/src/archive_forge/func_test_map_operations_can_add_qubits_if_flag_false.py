from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_can_add_qubits_if_flag_false():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q[0]))
    c_mapped = cirq.map_operations(c, lambda *_: cirq.CNOT(q[0], q[1]), raise_if_add_qubits=False)
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(cirq.CNOT(q[0], q[1])))