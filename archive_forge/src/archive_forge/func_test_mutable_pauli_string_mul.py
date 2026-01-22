import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_mul():
    a, b = cirq.LineQubit.range(2)
    p = cirq.X(a).mutable_copy()
    q = cirq.Y(b).mutable_copy()
    pq = cirq.X(a) * cirq.Y(b)
    assert p * q == pq
    assert isinstance(p * q, cirq.PauliString)
    assert 2 * p == cirq.X(a) * 2 == p * 2
    assert isinstance(p * 2, cirq.PauliString)
    assert isinstance(2 * p, cirq.PauliString)