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
def handle_run(self, record: Record) -> None:
    if self._settings._offline:
        self._run_proto = record.run
        result = proto_util._result_from_record(record)
        result.run_result.run.CopyFrom(record.run)
        self._respond_result(result)
    self._dispatch_record(record)