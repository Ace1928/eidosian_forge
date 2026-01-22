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
@_allow_deprecated_freezegun
@freezegun.freeze_time()
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_time_slots_async')
def test_get_schedule_time_filter_behavior(list_time_slots):
    list_time_slots.return_value = []
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    now = int(datetime.datetime.now().timestamp())
    in_two_weeks = int((datetime.datetime.now() + datetime.timedelta(weeks=2)).timestamp())
    processor.get_schedule()
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {in_two_weeks} AND end_time > {now}')
    with pytest.raises(ValueError, match='from_time of type'):
        processor.get_schedule(from_time=object())
    with pytest.raises(ValueError, match='to_time of type'):
        processor.get_schedule(to_time=object())
    processor.get_schedule(from_time=None, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', '')
    processor.get_schedule(from_time=datetime.timedelta(0), to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now}')
    processor.get_schedule(from_time=datetime.timedelta(seconds=200), to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now + 200}')
    test_timestamp = datetime.datetime.utcfromtimestamp(52)
    utc_ts = int(test_timestamp.timestamp())
    processor.get_schedule(from_time=test_timestamp, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {utc_ts}')
    processor.get_schedule(from_time=None, to_time=datetime.timedelta(0))
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {now}')
    processor.get_schedule(from_time=None, to_time=datetime.timedelta(seconds=200))
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {now + 200}')
    processor.get_schedule(from_time=None, to_time=test_timestamp)
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {utc_ts}')