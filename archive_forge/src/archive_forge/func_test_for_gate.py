import pytest
import cirq
def test_for_gate():

    class NoQidGate:

        def _qid_shape_(self):
            return ()

    class QuditGate:

        def _qid_shape_(self):
            return (4, 2, 3, 1)
    assert cirq.LineQid.for_gate(NoQidGate()) == []
    assert cirq.LineQid.for_gate(QuditGate()) == [cirq.LineQid(0, 4), cirq.LineQid(1, 2), cirq.LineQid(2, 3), cirq.LineQid(3, 1)]
    assert cirq.LineQid.for_gate(QuditGate(), start=5) == [cirq.LineQid(5, 4), cirq.LineQid(6, 2), cirq.LineQid(7, 3), cirq.LineQid(8, 1)]
    assert cirq.LineQid.for_gate(QuditGate(), step=2) == [cirq.LineQid(0, 4), cirq.LineQid(2, 2), cirq.LineQid(4, 3), cirq.LineQid(6, 1)]
    assert cirq.LineQid.for_gate(QuditGate(), start=5, step=-1) == [cirq.LineQid(5, 4), cirq.LineQid(4, 2), cirq.LineQid(3, 3), cirq.LineQid(2, 1)]