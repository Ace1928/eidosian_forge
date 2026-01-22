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
def handle_history(self, record: Record) -> None:
    history_dict = proto_util.dict_from_proto_list(record.history.item)
    if history_dict is not None:
        if '_runtime' not in history_dict:
            self._history_assign_runtime(record.history, history_dict)
    self._history_update(record.history, history_dict)
    self._dispatch_record(record)
    self._save_history(record.history)
    updated_keys = self._update_summary(history_dict)
    if updated_keys:
        updated_items = {k: self._consolidated_summary[k] for k in updated_keys}
        self._save_summary(updated_items)