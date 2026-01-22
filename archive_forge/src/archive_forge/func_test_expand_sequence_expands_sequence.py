from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_expand_sequence_expands_sequence():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q12 = cirq.GridQubit(1, 2)
    q13 = cirq.GridQubit(1, 3)
    q14 = cirq.GridQubit(1, 4)
    qubits = [q00, q01, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01]) == [q00, q10, q11, q01]
    qubits = [q00, q01, q02, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02]) == [q00, q10, q11, q01, q02]
    qubits = [q00, q01, q02, q11, q12]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02]) == [q00, q01, q11, q12, q02]
    qubits = [q00, q01, q02, q03, q11, q12]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03]) == [q00, q01, q11, q12, q02, q03]
    qubits = [q00, q01, q02, q03, q04, q10, q11, q13, q14]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03, q04]) == [q00, q10, q11, q01, q02, q03, q13, q14, q04]