import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_paulisum_validation():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    with pytest.raises(ValueError) as e:
        cirq.PauliSum([pstr1, pstr2])
    assert e.match('Consider using')
    with pytest.raises(ValueError):
        ld = cirq.LinearDict({pstr1: 2.0})
        cirq.PauliSum(ld)
    with pytest.raises(ValueError):
        key = frozenset([('q0', cirq.X)])
        ld = cirq.LinearDict({key: 2.0})
        cirq.PauliSum(ld)
    with pytest.raises(ValueError):
        key = frozenset([(q[0], cirq.H)])
        ld = cirq.LinearDict({key: 2.0})
        cirq.PauliSum(ld)
    key = frozenset([(q[0], cirq.X)])
    ld = cirq.LinearDict({key: 2.0})
    assert cirq.PauliSum(ld) == cirq.PauliSum.from_pauli_strings([2 * cirq.X(q[0])])
    ps = cirq.PauliSum()
    ps += cirq.I(cirq.LineQubit(0))
    assert ps == cirq.PauliSum(cirq.LinearDict({frozenset(): complex(1)}))