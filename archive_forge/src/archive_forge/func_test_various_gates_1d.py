import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_various_gates_1d():
    gate_op_cls = [cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.T]
    cross_gate_op_cls = [cirq.CNOT, cirq.SWAP]
    q0, q1 = cirq.LineQubit.range(2)
    for q0_gate_op in gate_op_cls:
        for q1_gate_op in gate_op_cls:
            for cross_gate_op in cross_gate_op_cls:
                circuit = cirq.Circuit(q0_gate_op(q0), q1_gate_op(q1), cross_gate_op(q0, q1))
                for initial_state in range(2 * 2):
                    assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1], initial_state=initial_state)