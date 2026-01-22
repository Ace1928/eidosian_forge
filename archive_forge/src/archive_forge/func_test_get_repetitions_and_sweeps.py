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
def test_get_repetitions_and_sweeps(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(run_context=util.pack_any(v2.run_context_pb2.RunContext(parameter_sweeps=[v2.run_context_pb2.ParameterSweep(repetitions=10)])))
    assert job.get_repetitions_and_sweeps() == (10, [cirq.UnitSweep])
    get_job.assert_called_once_with('a', 'b', 'steve', True)