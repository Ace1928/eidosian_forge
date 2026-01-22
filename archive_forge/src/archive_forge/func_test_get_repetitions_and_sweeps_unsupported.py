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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_async')
def test_get_repetitions_and_sweeps_unsupported(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(run_context=any_pb2.Any(type_url='type.googleapis.com/unknown.proto'))
    with pytest.raises(ValueError, match='unsupported run_context type: unknown.proto'):
        job.get_repetitions_and_sweeps()