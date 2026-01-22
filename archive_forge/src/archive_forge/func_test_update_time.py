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
def test_update_time(get_job):
    job = cg.EngineJob('a', 'b', 'steve', EngineContext())
    get_job.return_value = quantum.QuantumJob(update_time=timestamp_pb2.Timestamp(seconds=1581515101))
    assert job.update_time() == datetime.datetime(2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc)
    get_job.assert_called_once_with('a', 'b', 'steve', False)