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
def handle_request_pause(self, record: Record) -> None:
    if self._system_monitor is not None:
        logger.info('stopping system metrics thread')
        self._system_monitor.finish()
    if self._track_time is not None:
        self._accumulate_time += time.time() - self._track_time
        self._track_time = None