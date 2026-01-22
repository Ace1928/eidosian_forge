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
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.cancel_reservation_async')
def test_remove_reservation_cancel(cancel_reservation, get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = quantum.QuantumReservation(name=name, start_time=Timestamp(seconds=now + 10), end_time=Timestamp(seconds=now + 3610), whitelisted_users=['dstrain@google.com'])
    get_reservation.return_value = result
    cancel_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext(), quantum.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)))
    assert processor.remove_reservation('rid') == result
    cancel_reservation.assert_called_once_with('proj', 'p0', 'rid')