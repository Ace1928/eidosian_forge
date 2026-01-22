import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('shift,sign', itertools.product(range(3), (-1, +1)))
def test_pass_operations_over_single(shift: int, sign: int):
    q0, q1 = _make_qubits(2)
    X, Y, Z = (cirq.Pauli.by_relative_index(pauli, shift) for pauli in (cirq.X, cirq.Y, cirq.Z))
    op0 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before: cirq.PauliString[cirq.Qid] = cirq.PauliString({q0: X}, sign)
    ps_after = ps_before
    _assert_pass_over([op0], ps_before, ps_after)
    op0 = cirq.SingleQubitCliffordGate.from_pauli(X)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = ps_before
    _assert_pass_over([op0, op1], ps_before, ps_after)
    op0 = cirq.SingleQubitCliffordGate.from_double_map({Z: (X, False), X: (Z, False)})(q0)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: Z, q1: Y}, sign)
    _assert_pass_over([op0], ps_before, ps_after)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(X)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = -ps_before
    _assert_pass_over([op1], ps_before, ps_after)
    ps_after = cirq.PauliString({q0: Z, q1: Y}, -sign)
    _assert_pass_over([op0, op1], ps_before, ps_after)
    op0 = cirq.SingleQubitCliffordGate.from_pauli(Z, True)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(X, True)(q0)
    ps_before = cirq.PauliString({q0: X}, sign)
    ps_after = cirq.PauliString({q0: Y}, -sign)
    _assert_pass_over([op0, op1], ps_before, ps_after)