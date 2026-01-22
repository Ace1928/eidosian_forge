import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_all_or_any_terminal(circuit_cls):

    def is_x_pow_gate(op):
        return isinstance(op.gate, cirq.XPowGate)
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    ya = cirq.Y.on(a)
    yb = cirq.Y.on(b)
    c = circuit_cls()
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(ya)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(ya, yb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(ya, yb, xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(ya, yb, xa, xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xa, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xa, ya)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xb, ya, yb)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)
    c = circuit_cls(xa, ya, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    def is_circuit_op(op):
        isinstance(op, cirq.CircuitOperation)
    cop_1 = cirq.CircuitOperation(cirq.FrozenCircuit(xa, ya))
    cop_2 = cirq.CircuitOperation(cirq.FrozenCircuit(cop_1, xb))
    c = circuit_cls(cop_2, yb)
    assert c.are_all_matches_terminal(is_circuit_op)
    assert not c.are_any_matches_terminal(is_circuit_op)