from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_find_path_between_finds_path():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q12 = cirq.GridQubit(1, 2)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    q22 = cirq.GridQubit(2, 2)
    qubits = [q00, q01, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) == [q10, q11]
    qubits = [q00, q01, q02, q10, q20, q21, q22, q12]
    path_1 = [q00, q01, q02]
    path_2 = [q00, q10, q20, q21, q22, q12, q02]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) == path_2[1:-1]
    assert search._find_path_between(q02, q00, set(path_1)) == path_2[-2:0:-1]
    assert search._find_path_between(q00, q02, set(path_2)) == path_1[1:-1]
    assert search._find_path_between(q02, q00, set(path_2)) == path_1[-2:0:-1]