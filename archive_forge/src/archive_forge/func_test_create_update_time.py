import datetime
import pytest
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
def test_create_update_time():
    program = NothingProgram([cirq.Circuit()], None)
    create_time = datetime.datetime.fromtimestamp(1000)
    update_time = datetime.datetime.fromtimestamp(2000)
    program._create_time = create_time
    program._update_time = update_time
    assert program.create_time() == create_time
    assert program.update_time() == update_time