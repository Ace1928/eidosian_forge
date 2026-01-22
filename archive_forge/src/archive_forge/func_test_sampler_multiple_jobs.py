from unittest import mock
import pandas as pd
import sympy as sp
import cirq
import cirq_ionq as ionq
def test_sampler_multiple_jobs():
    mock_service = mock.MagicMock()
    job_dict0 = {'id': '1', 'status': 'completed', 'qubits': '1', 'target': 'qpu', 'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'}}
    job_dict1 = {'id': '1', 'status': 'completed', 'qubits': '1', 'target': 'qpu', 'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'}}
    job0 = ionq.Job(client=mock_service, job_dict=job_dict0)
    job1 = ionq.Job(client=mock_service, job_dict=job_dict1)
    mock_service.create_job.side_effect = [job0, job1]
    mock_service.get_results.side_effect = [{'0': '0.25', '1': '0.75'}, {'0': '0.5', '1': '0.5'}]
    sampler = ionq.Sampler(service=mock_service, timeout_seconds=10, target='qpu')
    q0 = cirq.LineQubit(0)
    x = sp.Symbol('x')
    circuit = cirq.Circuit(cirq.X(q0) ** x, cirq.measure(q0, key='a'))
    results = sampler.sample(program=circuit, repetitions=4, params=[cirq.ParamResolver({x: 0.5}), cirq.ParamResolver({x: 0.6})])
    pd.testing.assert_frame_equal(results, pd.DataFrame(columns=['x', 'a'], index=[0, 1, 2, 3] * 2, data=[[0.5, 0], [0.5, 1], [0.5, 1], [0.5, 1], [0.6, 0], [0.6, 0], [0.6, 1], [0.6, 1]]))
    circuit0 = cirq.Circuit(cirq.X(q0) ** 0.5, cirq.measure(q0, key='a'))
    circuit1 = cirq.Circuit(cirq.X(q0) ** 0.6, cirq.measure(q0, key='a'))
    mock_service.create_job.assert_has_calls([mock.call(circuit=circuit0, repetitions=4, target='qpu'), mock.call(circuit=circuit1, repetitions=4, target='qpu')])
    assert mock_service.create_job.call_count == 2