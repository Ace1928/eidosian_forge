from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('sym', [sympy.Symbol('x'), sympy.Symbol('x') + 1])
def test_blocked_by_unknown_and_symbols(sym):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.SWAP(a, b)], [cirq.X(a)]), expected=quick_circuit([cirq.X(a)], [cirq.SWAP(a, b)], [cirq.X(a)]))
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.Z(a) ** sym], [cirq.X(a)]), expected=quick_circuit([cirq.X(a)], [cirq.Z(a) ** sym], [cirq.X(a)]), compare_unitaries=False)
    assert_optimizes(before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** sym], [cirq.X(a)]), expected=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** sym], [cirq.X(a)]), compare_unitaries=False)