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
def test_receives_results_via_stream_returns_correct_results():
    qjob = quantum.QuantumJob(execution_status=quantum.ExecutionStatus(state=quantum.ExecutionStatus.State.SUCCESS), update_time=UPDATE_TIME)
    result_future = duet.completed_future(RESULTS)
    job = cg.EngineJob('a', 'b', 'steve', EngineContext(), _job=qjob, job_result_future=result_future)
    data = job.results()
    assert len(data) == 2
    assert str(data[0]) == 'q=0110'
    assert str(data[1]) == 'q=1010'