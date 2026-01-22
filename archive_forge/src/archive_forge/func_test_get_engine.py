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
@mock.patch('cirq_google.cloud.quantum.QuantumEngineServiceClient')
def test_get_engine(build):
    with mock.patch('google.auth.default', lambda *args, **kwargs: (None, 'project!')):
        eng = cirq_google.get_engine()
        assert eng.project_id == 'project!'
    with mock.patch('google.auth.default', lambda *args, **kwargs: (None, None)):
        with pytest.raises(EnvironmentError, match='GOOGLE_CLOUD_PROJECT'):
            _ = cirq_google.get_engine()
        _ = cirq_google.get_engine('project!')