import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_operates_on():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert not cirq.Moment().operates_on([])
    assert not cirq.Moment().operates_on([a])
    assert not cirq.Moment().operates_on([b])
    assert not cirq.Moment().operates_on([a, b])
    assert not cirq.Moment([cirq.X(a)]).operates_on([])
    assert cirq.Moment([cirq.X(a)]).operates_on([a])
    assert not cirq.Moment([cirq.X(a)]).operates_on([b])
    assert cirq.Moment([cirq.X(a)]).operates_on([a, b])
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on([])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([b])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, b])
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on([c])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, c])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, b, c])
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([b])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b])
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([c])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, c])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b, c])