from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_unroll_circuit_op_greedy_frontier_doesnt_touch_same_op_twice():
    q = cirq.NamedQubit('q')
    nested_ops = [cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q)))] * 5
    nested_circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit(nested_ops))
    c = cirq.Circuit(nested_circuit_op, nested_circuit_op, nested_circuit_op)
    c_expected = cirq.Circuit(nested_ops, nested_ops, nested_ops)
    c_unrolled = cirq.unroll_circuit_op_greedy_frontier(c, tags_to_check=None)
    cirq.testing.assert_same_circuits(c_unrolled, c_expected)