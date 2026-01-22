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
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_time_slots_async')
def test_get_schedule_filter_by_time_slot(list_time_slots):
    results = [quantum.QuantumTimeSlot(processor_name='potofgold', start_time=Timestamp(seconds=1000020000), end_time=Timestamp(seconds=1000040000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE, maintenance_config=quantum.QuantumTimeSlot.MaintenanceConfig(title='Testing', description='Testing some new configuration.'))]
    list_time_slots.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.get_schedule(datetime.datetime.fromtimestamp(1000000000), datetime.datetime.fromtimestamp(1000050000), quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE) == results
    list_time_slots.assert_called_once_with('proj', 'p0', 'start_time < 1000050000 AND end_time > 1000000000 AND ' + 'time_slot_type = MAINTENANCE')