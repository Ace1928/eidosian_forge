import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_dict_functionality():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString()
    with pytest.raises(KeyError):
        _ = p[a]
    assert p.get(a) is None
    assert a not in p
    assert not bool(p)
    p[a] = cirq.X
    assert bool(p)
    assert a in p
    assert p[a] == cirq.X
    p[a] = 'Y'
    assert p[a] == cirq.Y
    p[a] = 3
    assert p[a] == cirq.Z
    p[a] = 'I'
    assert a not in p
    p[a] = 0
    assert a not in p
    assert len(p) == 0
    p[b] = 'Y'
    p[a] = 'X'
    p[c] = 'Z'
    assert len(p) == 3
    assert list(iter(p)) == [b, a, c]
    assert list(p.values()) == [cirq.Y, cirq.X, cirq.Z]
    assert list(p.keys()) == [b, a, c]
    assert p.keys() == {a, b, c}
    assert p.keys() ^ {c} == {a, b}
    del p[b]
    assert b not in p