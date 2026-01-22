from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_find_sequence_assembles_head_and_tail():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    with mock.patch.object(search, '_sequence_search') as sequence_search:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]
        assert search._find_sequence() == qubits
        sequence_search.assert_has_calls([mock.call(start, []), mock.call(start, tail)])