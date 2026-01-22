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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_job_results_async')
def test_results_getitem(get_job_results):
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=UPDATE_TIME)
    get_job_results.return_value = RESULTS
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob)
    assert str(job[0]) == 'q=0110'
    assert str(job[1]) == 'q=1010'
    with pytest.raises(IndexError):
        _ = job[2]