import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient')
def test_create_context(client):
    with pytest.raises(ValueError, match='specify service_args and verbose or client'):
        EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'}, True, mock.Mock())
    with pytest.raises(ValueError, match='no longer supported'):
        _ = EngineContext(cg.engine.engine.ProtoVersion.V1, {'args': 'test'}, True)
    context = EngineContext(cg.engine.engine.ProtoVersion.V2, {'args': 'test'}, True)
    assert context.proto_version == cg.engine.engine.ProtoVersion.V2
    assert client.called_with({'args': 'test'}, True)
    assert context.copy().proto_version == context.proto_version
    assert context.copy().client == context.client
    assert context.copy() == context