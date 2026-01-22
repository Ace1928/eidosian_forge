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
def handle_request_attach(self, record: Record) -> None:
    result = proto_util._result_from_record(record)
    attach_id = record.request.attach.attach_id
    assert attach_id
    assert self._run_proto
    result.response.attach_response.run.CopyFrom(self._run_proto)
    self._respond_result(result)