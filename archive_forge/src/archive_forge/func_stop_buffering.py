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
def stop_buffering(self, kernel_id):
    """Stop buffering kernel messages

        Parameters
        ----------
        kernel_id : str
            The id of the kernel to stop buffering.
        """
    self.log.debug('Clearing buffer for %s', kernel_id)
    self._check_kernel_id(kernel_id)
    if kernel_id not in self._kernel_buffers:
        return
    buffer_info = self._kernel_buffers.pop(kernel_id)
    for stream in buffer_info['channels'].values():
        if not stream.socket.closed:
            stream.on_recv(None)
            stream.close()
    msg_buffer = buffer_info['buffer']
    if msg_buffer:
        self.log.info('Discarding %s buffered messages for %s', len(msg_buffer), buffer_info['session_key'])