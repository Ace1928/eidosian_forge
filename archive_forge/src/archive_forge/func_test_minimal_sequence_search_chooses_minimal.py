from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_minimal_sequence_search_chooses_minimal():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q10)
    assert search._choose_next_qubit(q10, {q10}) == q20
    assert search._choose_next_qubit(q20, {q10, q20}) == q21