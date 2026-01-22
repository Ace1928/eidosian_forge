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
def test_receives_job_via_stream_raises_and_updates_underlying_job():
    expected_error_code = quantum.ExecutionStatus.Failure.Code.SYSTEM_ERROR
    expected_error_message = 'system error'
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS, failure=quantum.ExecutionStatus.Failure(error_code=expected_error_code, error_message=expected_error_message)), update_time=UPDATE_TIME)
    result_future = duet.completed_future(qjob)
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future)
    qjob.execution_status.state = quantum.ExecutionStatus.State.FAILURE
    with pytest.raises(RuntimeError):
        job.results()
    actual_error_code, actual_error_message = job.failure()
    assert actual_error_code == expected_error_code.name
    assert actual_error_message == expected_error_message