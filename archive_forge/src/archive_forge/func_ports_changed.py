from __future__ import annotations
import asyncio
import os
import pathlib
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, wraps
from jupyter_client.ioloop.manager import AsyncIOLoopKernelManager
from jupyter_client.multikernelmanager import AsyncMultiKernelManager, MultiKernelManager
from jupyter_client.session import Session
from jupyter_core.paths import exists
from jupyter_core.utils import ensure_async
from jupyter_events import EventLogger
from jupyter_events.schema_registry import SchemaRegistryException
from overrides import overrides
from tornado import web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import (
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus.metrics import KERNEL_CURRENTLY_RUNNING_TOTAL
from jupyter_server.utils import ApiPath, import_item, to_os_path
def ports_changed(self, kernel_id):
    """Used by ZMQChannelsHandler to determine how to coordinate nudge and replays.

        Ports are captured when starting a kernel (via MappingKernelManager).  Ports
        are considered changed (following restarts) if the referenced KernelManager
        is using a set of ports different from those captured at startup.  If changes
        are detected, the captured set is updated and a value of True is returned.

        NOTE: Use is exclusive to ZMQChannelsHandler because this object is a singleton
        instance while ZMQChannelsHandler instances are per WebSocket connection that
        can vary per kernel lifetime.
        """
    changed_ports = self._get_changed_ports(kernel_id)
    if changed_ports:
        self.log.debug('Port change detected for kernel: %s', kernel_id)
        self._kernel_ports[kernel_id] = changed_ports
        return True
    return False