import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_once():
    with pytest.raises(TypeError, match='no _decompose_with_context_ or _decompose_ method'):
        _ = cirq.decompose_once(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once(DecomposeNotImplemented())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once(DecomposeNone())
    assert cirq.decompose_once(NoMethod(), 5) == 5
    assert cirq.decompose_once(DecomposeNotImplemented(), None) is None
    assert cirq.decompose_once(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.decompose_once(DecomposeNone(), 0) == 0
    op = cirq.X(cirq.NamedQubit('q'))
    assert cirq.decompose_once(DecomposeGiven(op)) == [op]
    assert cirq.decompose_once(DecomposeGiven([[[op]], []])) == [op]
    assert cirq.decompose_once(DecomposeGiven((op for _ in range(2)))) == [op, op]
    assert type(cirq.decompose_once(DecomposeGiven((op for _ in range(2))))) == list
    assert cirq.decompose_once(DecomposeGenerated()) == [cirq.X(cirq.LineQubit(0)), cirq.Y(cirq.LineQubit(1))]