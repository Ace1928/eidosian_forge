from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
@mock.patch('cirq_google.line.placement.greedy._PickLargestArea')
@mock.patch('cirq_google.line.placement.greedy._PickFewestNeighbors')
def test_greedy_search_method_calls_minimal_only(minimal, largest):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    device = _create_device([q00, q01])
    length = 2
    sequence = [q00, q01]
    minimal.return_value.get_or_search.return_value = sequence
    method = greedy.GreedySequenceSearchStrategy('minimal_connectivity')
    assert method.place_line(device, length) == GridQubitLineTuple(sequence)
    largest.return_value.get_or_search.assert_not_called()
    minimal.return_value.get_or_search.assert_called_once_with()