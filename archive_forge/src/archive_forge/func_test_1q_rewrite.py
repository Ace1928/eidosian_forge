from typing import List
import numpy as np
import pytest
import cirq
def test_1q_rewrite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.X(q1), cirq.CZ(q0, q1), cirq.Y(q1), cirq.measure(q0, q1))
    assert_optimizes(optimized=cirq.merge_k_qubit_unitaries(circuit, k=1, rewriter=lambda ops: cirq.H(ops.qubits[0])), expected=cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.CZ(q0, q1), cirq.H(q1), cirq.measure(q0, q1)))