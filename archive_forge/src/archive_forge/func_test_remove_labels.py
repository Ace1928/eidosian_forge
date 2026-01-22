import datetime
from unittest import mock
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient.remove_job_labels_async')
def test_remove_labels(remove_job_labels):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=quantum.QuantumJob(labels={'a': '1', 'b': '1'}))
    assert job.labels() == {'a': '1', 'b': '1'}
    remove_job_labels.return_value = quantum.QuantumJob(labels={'b': '1'})
    assert job.remove_labels(['a']).labels() == {'b': '1'}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a'])
    remove_job_labels.return_value = quantum.QuantumJob(labels={})
    assert job.remove_labels(['a', 'b', 'c']).labels() == {}
    remove_job_labels.assert_called_with('a', 'b', 'steve', ['a', 'b', 'c'])