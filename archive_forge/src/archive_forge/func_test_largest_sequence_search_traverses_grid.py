from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_largest_sequence_search_traverses_grid():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q20 = cirq.GridQubit(2, 0)
    q30 = cirq.GridQubit(3, 0)
    q40 = cirq.GridQubit(4, 0)
    q41 = cirq.GridQubit(4, 1)
    q42 = cirq.GridQubit(4, 2)
    q50 = cirq.GridQubit(5, 0)
    qubits = [q00, q10, q11, q20, q30, q40, q50, q41, q42]
    device = _create_device(qubits)
    search = greedy._PickLargestArea(device, q20)
    assert search._choose_next_qubit(q20, {q20}) == q30
    assert search._choose_next_qubit(q30, {q20, q30}) == q40
    assert search._choose_next_qubit(q40, {q20, q30, q40}) == q41
    assert search._choose_next_qubit(q41, {q20, q30, q40, q41}) == q42
    assert search._choose_next_qubit(q42, {q20, q30, q40, q41, q42}) is None
    method = greedy.GreedySequenceSearchStrategy('largest_area')
    assert method.place_line(device, 7) == GridQubitLineTuple([q00, q10, q20, q30, q40, q41, q42])
    with pytest.raises(NotFoundError):
        _ = method.place_line(device, 8)