import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_operation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.Moment().with_operation(cirq.X(a)) == cirq.Moment([cirq.X(a)])
    assert cirq.Moment([cirq.X(a)]).with_operation(cirq.X(b)) == cirq.Moment([cirq.X(a), cirq.X(b)])
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a)]).with_operation(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operation(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operation(cirq.X(b))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operation(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operation(cirq.X(b))