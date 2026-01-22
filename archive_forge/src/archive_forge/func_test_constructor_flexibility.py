import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_constructor_flexibility():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.PauliString(cirq.CZ(a, b))
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.PauliString('test')
    with pytest.raises(TypeError, match='S is not a Pauli'):
        _ = cirq.PauliString(qubit_pauli_map={a: cirq.S})
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.PauliString(cirq.Z(a) + cirq.Z(b))
    assert cirq.PauliString(cirq.X(a)) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([cirq.X(a)]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.X(a)]]]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.I(a)]]]) == cirq.PauliString()
    assert cirq.PauliString(1, 2, 3, cirq.X(a), cirq.Y(a)) == cirq.PauliString(qubit_pauli_map={a: cirq.Z}, coefficient=6j)
    assert cirq.PauliString(cirq.X(a), cirq.X(a)) == cirq.PauliString()
    assert cirq.PauliString(cirq.X(a), cirq.X(b)) == cirq.PauliString(qubit_pauli_map={a: cirq.X, b: cirq.X})
    assert cirq.PauliString(0) == cirq.PauliString(coefficient=0)
    assert cirq.PauliString(1, 2, 3, {a: cirq.X}, cirq.Y(a)) == cirq.PauliString(qubit_pauli_map={a: cirq.Z}, coefficient=6j)