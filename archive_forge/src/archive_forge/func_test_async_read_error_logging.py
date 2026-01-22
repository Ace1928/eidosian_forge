from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
@skipPypy3V58
@gen_test
def test_async_read_error_logging(self):
    server, client = (yield self.make_iostream_pair())
    closed = Event()
    server.set_close_callback(closed.set)
    try:
        server.read_bytes(1)
        client.write(b'a')

        def fake_read_from_fd():
            os.close(server.socket.fileno())
            server.__class__.read_from_fd(server)
        server.read_from_fd = fake_read_from_fd
        with ExpectLog(gen_log, 'error on read'):
            yield closed.wait()
    finally:
        server.close()
        client.close()