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
def handle_request_partial_history(self, record: Record) -> None:
    partial_history = record.request.partial_history
    flush = None
    if partial_history.HasField('action'):
        flush = partial_history.action.flush
    step = None
    if partial_history.HasField('step'):
        step = partial_history.step.num
    history_dict = proto_util.dict_from_proto_list(partial_history.item)
    if step is not None:
        if step < self._step:
            if not self._dropped_history:
                message = f'Step only supports monotonically increasing values, use define_metric to set a custom x axis. For details see: {wburls.wburls.get('wandb_define_metric')}'
                self._internal_messages.warning.append(message)
                self._dropped_history = True
            message = f'(User provided step: {step} is less than current step: {self._step}. Dropping entry: {history_dict}).'
            self._internal_messages.warning.append(message)
            return
        elif step > self._step:
            self._flush_partial_history()
            self._step = step
    elif flush is None:
        flush = True
    self._partial_history.update(history_dict)
    if flush:
        self._flush_partial_history(self._step)