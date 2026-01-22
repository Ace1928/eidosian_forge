import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_indexes_by_qubit():
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.H(a), cirq.CNOT(b, c)])
    assert moment[a] == cirq.H(a)
    assert moment[b] == cirq.CNOT(b, c)
    assert moment[c] == cirq.CNOT(b, c)