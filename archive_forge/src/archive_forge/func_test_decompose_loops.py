import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_decompose_loops():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    base_op = cirq.CircuitOperation(circuit)
    op = base_op.with_qubits(b, a).repeat(3)
    expected_circuit = cirq.Circuit(cirq.H(b), cirq.CX(b, a), cirq.H(b), cirq.CX(b, a), cirq.H(b), cirq.CX(b, a))
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit
    op = base_op.repeat(-2)
    expected_circuit = cirq.Circuit(cirq.CX(a, b), cirq.H(a), cirq.CX(a, b), cirq.H(a))
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit