import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_throws_when_indexed_by_unused_qubit():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment([cirq.H(a)])
    with pytest.raises(KeyError, match="Moment doesn't act on given qubit"):
        _ = moment[b]