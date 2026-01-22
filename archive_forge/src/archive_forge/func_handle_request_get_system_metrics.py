import json
import logging
import math
import numbers
import time
from collections import defaultdict
from queue import Queue
from threading import Event
from typing import (
from wandb.proto.wandb_internal_pb2 import (
from ..interface.interface_queue import InterfaceQueue
from ..lib import handler_util, proto_util, tracelog, wburls
from . import context, sample, tb_watcher
from .settings_static import SettingsStatic
from .system.system_monitor import SystemMonitor
def handle_request_get_system_metrics(self, record: Record) -> None:
    result = proto_util._result_from_record(record)
    if self._system_monitor is None:
        return
    buffer = self._system_monitor.buffer
    for key, samples in buffer.items():
        buff = []
        for s in samples:
            sms = SystemMetricSample()
            sms.timestamp.FromMicroseconds(int(s[0] * 1000000.0))
            sms.value = s[1]
            buff.append(sms)
        result.response.get_system_metrics_response.system_metrics[key].CopyFrom(SystemMetricsBuffer(record=buff))
    self._respond_result(result)