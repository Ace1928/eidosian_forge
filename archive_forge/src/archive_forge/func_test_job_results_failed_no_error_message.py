from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_results_failed_no_error_message():
    job_dict = {'id': 'my_id', 'status': 'failed', 'failure': {}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match='failed'):
        _ = job.results()
    assert job.status() == 'failed'