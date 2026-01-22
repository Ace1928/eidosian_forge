from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_largest_collect_stops_on_used():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q05 = cirq.GridQubit(0, 5)
    q11 = cirq.GridQubit(1, 1)
    q14 = cirq.GridQubit(1, 4)
    q24 = cirq.GridQubit(2, 4)
    qubits = [q00, q01, q11, q02, q03, q04, q05, q14, q24]
    start = q02
    search = greedy._PickLargestArea(_create_device(qubits), start)
    assert search._collect_unused(start, {start, q04}) == {q00, q01, q11, q02, q03}