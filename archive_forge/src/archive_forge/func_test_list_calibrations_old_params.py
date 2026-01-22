from unittest import mock
import datetime
import duet
import pytest
import freezegun
import numpy as np
from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext
from cirq_google.cloud import quantum
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_calibrations_async')
def test_list_calibrations_old_params(list_calibrations):
    list_calibrations.return_value = [_CALIBRATION]
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    with cirq.testing.assert_deprecated('Change earliest_timestamp_seconds', deadline='v1.0'):
        assert [c.timestamp for c in processor.list_calibrations(earliest_timestamp_seconds=1562500000)] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', 'timestamp >= 1562500000')
    with cirq.testing.assert_deprecated('Change latest_timestamp_seconds', deadline='v1.0'):
        assert [c.timestamp for c in processor.list_calibrations(latest_timestamp_seconds=1562600000)] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', 'timestamp <= 1562600000')