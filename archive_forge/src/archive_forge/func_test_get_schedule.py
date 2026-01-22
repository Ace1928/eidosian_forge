import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_get_schedule():
    time_slot = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=1000000), end_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(1500000), to_time=_time(2500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(1500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(750000)) == []
    assert p.get_schedule(from_time=_time(2500000), to_time=_time(300000)) == []
    unbounded_start = quantum.QuantumTimeSlot(processor_name='test', end_time=Timestamp(seconds=1000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    unbounded_end = quantum.QuantumTimeSlot(processor_name='test', start_time=Timestamp(seconds=2000000), time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    p = NothingProcessor(processor_id='test', schedule=[unbounded_start, unbounded_end])
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [unbounded_start, unbounded_end]
    assert p.get_schedule(from_time=_time(1500000), to_time=_time(2500000)) == [unbounded_end]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(1500000)) == [unbounded_start]
    assert p.get_schedule(from_time=_time(1200000), to_time=_time(1500000)) == []