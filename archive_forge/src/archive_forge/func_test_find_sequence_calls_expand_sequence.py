from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_find_sequence_calls_expand_sequence():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    with mock.patch.object(search, '_sequence_search') as sequence_search, mock.patch.object(search, '_expand_sequence') as expand_sequence:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]
        search._find_sequence()
        expand_sequence.assert_called_once_with(qubits)