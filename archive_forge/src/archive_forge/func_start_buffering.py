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
def start_buffering(self, kernel_id, session_key, channels):
    """Start buffering messages for a kernel

        Parameters
        ----------
        kernel_id : str
            The id of the kernel to stop buffering.
        session_key : str
            The session_key, if any, that should get the buffer.
            If the session_key matches the current buffered session_key,
            the buffer will be returned.
        channels : dict({'channel': ZMQStream})
            The zmq channels whose messages should be buffered.
        """
    if not self.buffer_offline_messages:
        for _, stream in channels.items():
            stream.close()
        return
    self.log.info('Starting buffering for %s', session_key)
    self._check_kernel_id(kernel_id)
    self.stop_buffering(kernel_id)
    buffer_info = self._kernel_buffers[kernel_id]
    buffer_info['session_key'] = session_key
    buffer_info['buffer'] = []
    buffer_info['channels'] = channels

    def buffer_msg(channel, msg_parts):
        self.log.debug('Buffering msg on %s:%s', kernel_id, channel)
        buffer_info['buffer'].append((channel, msg_parts))
    for channel, stream in channels.items():
        stream.on_recv(partial(buffer_msg, channel))