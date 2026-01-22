import datetime
import pytest
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
def test_description_and_labels():
    program = NothingProgram([cirq.Circuit()], None)
    assert not program.description()
    program.set_description('nothing much')
    assert program.description() == 'nothing much'
    program.set_description('other desc')
    assert program.description() == 'other desc'
    assert program.labels() == {}
    program.set_labels({'key': 'green'})
    assert program.labels() == {'key': 'green'}
    program.add_labels({'door': 'blue', 'curtains': 'white'})
    assert program.labels() == {'key': 'green', 'door': 'blue', 'curtains': 'white'}
    program.remove_labels(['key', 'door'])
    assert program.labels() == {'curtains': 'white'}
    program.set_labels({'walls': 'gray'})
    assert program.labels() == {'walls': 'gray'}