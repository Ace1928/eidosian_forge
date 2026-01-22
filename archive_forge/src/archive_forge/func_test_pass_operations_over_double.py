import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('shift,t_or_f1, t_or_f2,neg', itertools.product(range(3), *((True, False),) * 3))
def test_pass_operations_over_double(shift: int, t_or_f1: bool, t_or_f2: bool, neg: bool):
    sign = -1 if neg else +1
    q0, q1, q2 = _make_qubits(3)
    X, Y, Z = (cirq.Pauli.by_relative_index(pauli, shift) for pauli in (cirq.X, cirq.Y, cirq.Z))
    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString(qubit_pauli_map={q0: Z, q2: Y}, coefficient=sign)
    ps_after = cirq.PauliString(qubit_pauli_map={q0: Z, q2: Y}, coefficient=sign)
    _assert_pass_over([op0], ps_before, ps_after)
    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q2: Y}, sign)
    ps_after = cirq.PauliString({q0: Z, q2: Y, q1: X}, sign)
    _assert_pass_over([op0], ps_before, ps_after)
    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q1: Y}, sign)
    _assert_pass_over([op0], ps_before, ps_after)
    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: X, q1: Z}, -1 if neg ^ t_or_f1 ^ t_or_f2 else +1)
    _assert_pass_over([op0], ps_before, ps_after)
    op0 = cirq.PauliInteractionGate(X, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: Y, q1: Z}, +1 if neg ^ t_or_f1 ^ t_or_f2 else -1)
    _assert_pass_over([op0], ps_before, ps_after)