import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_various_gates_2d():
    gate_op_cls = [cirq.I, cirq.H]
    cross_gate_op_cls = [cirq.CNOT, cirq.SWAP]
    q0, q1, q2, q3, q4, q5 = cirq.GridQubit.rect(3, 2)
    for q0_gate_op in gate_op_cls:
        for q1_gate_op in gate_op_cls:
            for q2_gate_op in gate_op_cls:
                for q3_gate_op in gate_op_cls:
                    for cross_gate_op1 in cross_gate_op_cls:
                        for cross_gate_op2 in cross_gate_op_cls:
                            circuit = cirq.Circuit(q0_gate_op(q0), q1_gate_op(q1), cross_gate_op1(q0, q1), q2_gate_op(q2), q3_gate_op(q3), cross_gate_op2(q3, q1))
                            assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1, q2, q3, q4, q5])