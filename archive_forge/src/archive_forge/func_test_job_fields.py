from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_fields():
    job_dict = {'id': 'my_id', 'target': 'qpu', 'name': 'bacon', 'qubits': '5', 'status': 'completed', 'metadata': {'shots': 1000, 'measurement0': f'a{chr(31)}0,1'}}
    job = ionq.Job(None, job_dict)
    assert job.job_id() == 'my_id'
    assert job.target() == 'qpu'
    assert job.name() == 'bacon'
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
    assert job.measurement_dict() == {'a': [0, 1]}