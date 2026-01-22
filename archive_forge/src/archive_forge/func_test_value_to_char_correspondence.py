import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_value_to_char_correspondence():
    d = cirq.DensePauliString
    assert [d.I_VAL, d.X_VAL, d.Y_VAL, d.Z_VAL] == [0, 1, 2, 3]
    assert list(d([cirq.I, cirq.X, cirq.Y, cirq.Z]).pauli_mask) == [0, 1, 2, 3]
    assert list(d('IXYZ').pauli_mask) == [0, 1, 2, 3]
    assert list(d([d.I_VAL, d.X_VAL, d.Y_VAL, d.Z_VAL]).pauli_mask) == [0, 1, 2, 3]
    assert d('Y') * d('Z') == 1j * d('X')
    assert d('Z') * d('X') == 1j * d('Y')
    assert d('X') * d('Y') == 1j * d('Z')
    assert d('Y') * d('X') == -1j * d('Z')
    assert d('X') * d('Z') == -1j * d('Y')
    assert d('Z') * d('Y') == -1j * d('X')