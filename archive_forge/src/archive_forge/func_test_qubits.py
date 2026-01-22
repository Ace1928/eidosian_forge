import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).qubits == {a, b}
    assert cirq.Moment([cirq.X(a)]).qubits == {a}
    assert cirq.Moment([cirq.CZ(a, b)]).qubits == {a, b}