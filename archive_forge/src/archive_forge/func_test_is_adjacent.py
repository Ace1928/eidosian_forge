import pytest
import cirq
def test_is_adjacent():
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(2))
    assert cirq.LineQubit(1).is_adjacent(cirq.LineQubit(0))
    assert cirq.LineQubit(2).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(1).is_adjacent(cirq.LineQubit(3))
    assert not cirq.LineQubit(2).is_adjacent(cirq.LineQubit(0))
    assert cirq.LineQubit(2).is_adjacent(cirq.LineQid(3, 3))
    assert not cirq.LineQubit(2).is_adjacent(cirq.LineQid(0, 3))