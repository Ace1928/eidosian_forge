from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_cancel():
    ready_job = {'id': 'my_id', 'status': 'ready'}
    canceled_job = {'id': 'my_id', 'status': 'canceled'}
    mock_client = mock.MagicMock()
    mock_client.cancel_job.return_value = canceled_job
    job = ionq.Job(mock_client, ready_job)
    job.cancel()
    mock_client.cancel_job.assert_called_with(job_id='my_id')
    assert job.status() == 'canceled'