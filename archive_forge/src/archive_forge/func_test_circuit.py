import datetime
import pytest
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
def test_circuit():
    circuit1 = cirq.Circuit(cirq.X(cirq.LineQubit(1)))
    circuit2 = cirq.Circuit(cirq.Y(cirq.LineQubit(2)))
    program = NothingProgram([circuit1], None)
    assert program.batch_size() == 1
    assert program.get_circuit() == circuit1
    assert program.get_circuit(0) == circuit1
    assert program.batch_size() == 1
    program = NothingProgram([circuit1, circuit2], None)
    assert program.batch_size() == 2
    assert program.get_circuit(0) == circuit1
    assert program.get_circuit(1) == circuit2