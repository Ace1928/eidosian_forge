from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_results_rounding_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.0006', '2': '0.9994'}
    job_dict = {'id': 'my_id', 'status': 'completed', 'qubits': '2', 'target': 'qpu', 'metadata': {'shots': 5000, 'measurement0': f'a{chr(31)}0,1'}}
    job = ionq.Job(mock_client, job_dict)
    expected = ionq.QPUResult({0: 3, 1: 4997}, 2, {'a': [0, 1]})
    results = job.results()
    assert results == expected