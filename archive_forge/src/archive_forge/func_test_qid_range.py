import pytest
import cirq
def test_qid_range():
    assert cirq.LineQid.range(0, dimension=3) == []
    assert cirq.LineQid.range(1, dimension=3) == [cirq.LineQid(0, 3)]
    assert cirq.LineQid.range(2, dimension=3) == [cirq.LineQid(0, 3), cirq.LineQid(1, 3)]
    assert cirq.LineQid.range(5, dimension=3) == [cirq.LineQid(0, 3), cirq.LineQid(1, 3), cirq.LineQid(2, 3), cirq.LineQid(3, 3), cirq.LineQid(4, 3)]
    assert cirq.LineQid.range(0, 0, dimension=4) == []
    assert cirq.LineQid.range(0, 1, dimension=4) == [cirq.LineQid(0, 4)]
    assert cirq.LineQid.range(1, 4, dimension=4) == [cirq.LineQid(1, 4), cirq.LineQid(2, 4), cirq.LineQid(3, 4)]
    assert cirq.LineQid.range(3, 1, -1, dimension=1) == [cirq.LineQid(3, 1), cirq.LineQid(2, 1)]
    assert cirq.LineQid.range(3, 5, -1, dimension=2) == []
    assert cirq.LineQid.range(1, 5, 2, dimension=2) == [cirq.LineQid(1, 2), cirq.LineQid(3, 2)]