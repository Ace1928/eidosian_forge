from unittest import mock
import warnings
import pytest
import cirq_ionq as ionq
def test_job_results_simulator_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job_dict = {'id': 'my_id', 'status': 'completed', 'qubits': '2', 'target': 'simulator', 'metadata': {'shots': '100'}}
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {}, 100)