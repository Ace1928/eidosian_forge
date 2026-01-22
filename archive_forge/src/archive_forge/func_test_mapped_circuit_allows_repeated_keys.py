import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_mapped_circuit_allows_repeated_keys():
    q = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q, key='A')))
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    circuit = op2.mapped_circuit(deep=True)
    cirq.testing.assert_has_diagram(circuit, "0: ───M('A')───M('A')───", use_unicode_characters=True)
    op1 = cirq.measure(q, key='A')
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    circuit = op2.mapped_circuit()
    cirq.testing.assert_has_diagram(circuit, "0: ───M('A')───M('A')───", use_unicode_characters=True)