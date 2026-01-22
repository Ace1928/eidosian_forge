import asyncio
import datetime
import os
from unittest import mock
import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_remove_job_labels(client_constructor):
    grpc_client = _setup_client_mock(client_constructor)
    existing = quantum.QuantumJob(labels={'color': 'red', 'weather': 'sun', 'run': '1'}, label_fingerprint='hash')
    grpc_client.get_quantum_job.return_value = existing
    result = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
    grpc_client.update_quantum_job.return_value = result
    client = EngineClient()
    assert client.remove_job_labels('proj', 'prog', 'job0', ['other']) == existing
    assert grpc_client.update_quantum_program.call_count == 0
    assert client.remove_job_labels('proj', 'prog', 'job0', ['hello', 'weather']) == result
    grpc_client.update_quantum_job.assert_called_with(quantum.UpdateQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', labels={'color': 'red', 'run': '1'}, label_fingerprint='hash'), update_mask=FieldMask(paths=['labels'])))
    assert client.remove_job_labels('proj', 'prog', 'job0', ['color', 'weather', 'run']) == result
    grpc_client.update_quantum_job.assert_called_with(quantum.UpdateQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0', quantum_job=quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0', label_fingerprint='hash'), update_mask=FieldMask(paths=['labels'])))