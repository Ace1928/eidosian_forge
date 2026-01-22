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
def handle_request_defer(self, record: Record) -> None:
    defer = record.request.defer
    state = defer.state
    logger.info(f'handle defer: {state}')
    if state == defer.FLUSH_STATS:
        if self._system_monitor is not None:
            self._system_monitor.finish()
    elif state == defer.FLUSH_TB:
        if self._tb_watcher:
            self._tb_watcher.finish()
            self._tb_watcher = None
    elif state == defer.FLUSH_PARTIAL_HISTORY:
        self._flush_partial_history()
    elif state == defer.FLUSH_SUM:
        self._save_summary(self._consolidated_summary, flush=True)
    self._dispatch_record(record, always_send=True)