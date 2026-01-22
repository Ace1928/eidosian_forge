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
def test_findall_operations_until_blocked(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)
    assert_findall_operations_until_blocked_as_expected()
    circuit = circuit_cls(cirq.H(a), cirq.CZ(a, b), cirq.H(b), cirq.CZ(b, c), cirq.H(c), cirq.CZ(c, d), cirq.H(d), cirq.CZ(c, d), cirq.H(c), cirq.CZ(b, c), cirq.H(b), cirq.CZ(a, b), cirq.H(a))
    expected_diagram = '\n0: ───H───@───────────────────────────────────────@───H───\n          │                                       │\n1: ───────@───H───@───────────────────────@───H───@───────\n                  │                       │\n2: ───────────────@───H───@───────@───H───@───────────────\n                          │       │\n3: ───────────────────────@───H───@───────────────────────\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    go_to_end = lambda op: False
    stop_if_op = lambda op: True
    stop_if_h_on_a = lambda op: op.gate == cirq.H and a in op.qubits
    assert_findall_operations_until_blocked_as_expected(is_blocker=go_to_end, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, is_blocker=go_to_end, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(start_frontier={a: 5}, is_blocker=stop_if_op, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(start_frontier={a: -100}, is_blocker=stop_if_op, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier={a: 100}, is_blocker=stop_if_op, expected_ops=[])
    for idx in range(15):
        for q in (a, b, c, d):
            assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier={q: idx}, is_blocker=stop_if_op, expected_ops=[])
        assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier={a: idx, b: idx, c: idx, d: idx}, is_blocker=stop_if_op, expected_ops=[])
    a_ending_ops = [(11, cirq.CZ.on(a, b)), (12, cirq.H.on(a))]
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier={a: idx}, is_blocker=go_to_end, expected_ops=a_ending_ops)
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier={a: idx}, is_blocker=stop_if_h_on_a, expected_ops=[(11, cirq.CZ.on(a, b))])
    circuit = circuit_cls([cirq.CZ(a, b), cirq.CZ(a, b), cirq.CZ(b, c)])
    expected_diagram = '\n0: ───@───@───────\n      │   │\n1: ───@───@───@───\n              │\n2: ───────────@───\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    start_frontier = {a: 0, b: 0}
    is_blocker = lambda next_op: sorted(next_op.qubits) != [a, b]
    expected_ops = [(0, cirq.CZ(a, b)), (1, cirq.CZ(a, b))]
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=expected_ops)
    circuit = circuit_cls([cirq.ZZ(a, b), cirq.ZZ(b, c)])
    expected_diagram = '\n0: ───ZZ────────\n      │\n1: ───ZZ───ZZ───\n           │\n2: ────────ZZ───\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    start_frontier = {a: 0, b: 0, c: 0}
    is_blocker = lambda op: a in op.qubits
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=[])
    circuit = circuit_cls([cirq.ZZ(a, b), cirq.XX(c, d), cirq.ZZ(b, c), cirq.Z(b)])
    expected_diagram = '\n0: ───ZZ────────────\n      │\n1: ───ZZ───ZZ───Z───\n           │\n2: ───XX───ZZ───────\n      │\n3: ───XX────────────\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    start_frontier = {a: 0, b: 0, c: 0, d: 0}
    is_blocker = lambda op: isinstance(op.gate, cirq.XXPowGate)
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=[(0, cirq.ZZ(a, b))])
    circuit = circuit_cls([cirq.XX(a, b), cirq.Z(a), cirq.ZZ(b, c), cirq.ZZ(c, d), cirq.Z(d)])
    expected_diagram = '\n0: ───XX───Z─────────────\n      │\n1: ───XX───ZZ────────────\n           │\n2: ────────ZZ───ZZ───────\n                │\n3: ─────────────ZZ───Z───\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    start_frontier = {a: 0, d: 0}
    assert_findall_operations_until_blocked_as_expected(circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=[])