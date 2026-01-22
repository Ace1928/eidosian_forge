import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_container_methods():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = cirq.Moment([cirq.H(a), cirq.H(b)])
    assert list(m) == list(m.operations)
    assert list(iter(m)) == list(m.operations)
    assert cirq.H(b) in m
    assert len(m) == 2