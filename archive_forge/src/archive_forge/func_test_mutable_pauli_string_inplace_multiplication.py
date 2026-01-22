import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_inplace_multiplication():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString()
    original = p
    p *= cirq.X(a)
    assert p == cirq.X(a) and p is original
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        p.inplace_left_multiply_by([cirq.X(a), cirq.CZ(a, b), cirq.Z(b)])
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        p.inplace_left_multiply_by(cirq.CZ(a, b))
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        p.inplace_right_multiply_by([cirq.X(a), cirq.CZ(a, b), cirq.Z(b)])
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        p.inplace_right_multiply_by(cirq.CZ(a, b))
    assert p == cirq.X(a) and p is original
    p *= cirq.Y(a)
    assert p == -1j * cirq.Z(a) and p is original
    p *= cirq.Y(a)
    assert p == cirq.X(a) and p is original
    p.inplace_left_multiply_by(cirq.Y(a))
    assert p == 1j * cirq.Z(a) and p is original
    p.inplace_left_multiply_by(cirq.Y(a))
    assert p == cirq.X(a) and p is original
    p.inplace_right_multiply_by(cirq.Y(a))
    assert p == -1j * cirq.Z(a) and p is original
    p.inplace_right_multiply_by(cirq.Y(a))
    assert p == cirq.X(a) and p is original
    p *= -1 * cirq.X(a) * cirq.X(b)
    assert p == -cirq.X(b) and p is original
    p.inplace_left_multiply_by({c: 'Z'})
    assert p == -cirq.X(b) * cirq.Z(c) and p is original
    p.inplace_right_multiply_by({c: 'Z'})
    assert p == -cirq.X(b) and p is original