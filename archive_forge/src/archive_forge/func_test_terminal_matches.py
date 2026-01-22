import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_terminal_matches():
    a, b = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(cirq.H(a), cirq.measure(b, key='m1'))
    op = cirq.CircuitOperation(fc)
    c = cirq.Circuit(cirq.X(a), op)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(cirq.X(b), op)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(cirq.measure(a), op)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(cirq.measure(b), op)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(op, cirq.X(a))
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(op, cirq.X(b))
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    c = cirq.Circuit(op, cirq.measure(a))
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = cirq.Circuit(op, cirq.measure(b))
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()