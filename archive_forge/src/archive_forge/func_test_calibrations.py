from typing import List
import datetime
import pytest
import numpy as np
import sympy
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor, VALID_LANGUAGES
def test_calibrations():
    now = datetime.datetime.now()
    future = int((datetime.datetime.now() + datetime.timedelta(hours=2)).timestamp())
    cal_proto1 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=10000)
    cal_proto2 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=20000)
    cal_proto3 = v2.metrics_pb2.MetricsSnapshot(timestamp_ms=future * 1000)
    cal1 = cirq_google.Calibration(cal_proto1)
    cal2 = cirq_google.Calibration(cal_proto2)
    cal3 = cirq_google.Calibration(cal_proto3)
    proc = SimulatedLocalProcessor(processor_id='test_proc', calibrations={10000: cal1, 20000: cal2, future: cal3})
    assert proc.get_calibration(10000) == cal1
    assert proc.get_calibration(20000) == cal2
    assert proc.get_calibration(future) == cal3
    assert proc.get_current_calibration() == cal2
    assert proc.list_calibrations(earliest_timestamp=5000, latest_timestamp=15000) == [cal1]
    assert proc.list_calibrations(earliest_timestamp=15000, latest_timestamp=25000) == [cal2]
    assert proc.list_calibrations(earliest_timestamp=now, latest_timestamp=now + datetime.timedelta(hours=2)) == [cal3]
    assert proc.list_calibrations(earliest_timestamp=datetime.date.today(), latest_timestamp=now + datetime.timedelta(hours=2)) == [cal3]
    cal_list = proc.list_calibrations(latest_timestamp=25000)
    assert len(cal_list) == 2
    assert cal1 in cal_list
    assert cal2 in cal_list
    cal_list = proc.list_calibrations(earliest_timestamp=15000)
    assert len(cal_list) == 2
    assert cal2 in cal_list
    assert cal3 in cal_list
    cal_list = proc.list_calibrations()
    assert len(cal_list) == 3
    assert cal1 in cal_list
    assert cal2 in cal_list
    assert cal3 in cal_list