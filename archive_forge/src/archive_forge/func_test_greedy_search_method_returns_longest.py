from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_greedy_search_method_returns_longest():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    device = _create_device([q00, q10])
    length = 1
    method = greedy.GreedySequenceSearchStrategy()
    assert method.place_line(device, length) == GridQubitLineTuple([q00])