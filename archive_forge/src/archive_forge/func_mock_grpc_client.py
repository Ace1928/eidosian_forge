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
@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch('cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient') as _fixture:
        yield _fixture