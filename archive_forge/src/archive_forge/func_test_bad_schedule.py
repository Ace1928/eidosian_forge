import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_bad_schedule():
    time_slot1 = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=3000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    time_slot2 = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=2000000), end_time=Timestamp(seconds=4000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    with pytest.raises(ValueError, match='cannot overlap'):
        _ = NothingProcessor(processor_id='test', schedule=[time_slot1, time_slot2])