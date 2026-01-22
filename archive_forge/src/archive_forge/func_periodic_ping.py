import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
def periodic_ping(self) -> None:
    """Send a ping to keep the websocket alive

        Called periodically if the websocket_ping_interval is set and non-zero.
        """
    if self.is_closing() and self.ping_callback is not None:
        self.ping_callback.stop()
        return
    now = IOLoop.current().time()
    since_last_pong = now - self.last_pong
    since_last_ping = now - self.last_ping
    assert self.ping_interval is not None
    assert self.ping_timeout is not None
    if since_last_ping < 2 * self.ping_interval and since_last_pong > self.ping_timeout:
        self.close()
        return
    self.write_ping(b'')
    self.last_ping = now