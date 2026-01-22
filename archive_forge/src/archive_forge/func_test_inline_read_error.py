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
@skipIfNonUnix
@skipPypy3V58
@gen_test
def test_inline_read_error(self: typing.Any):
    io_loop = IOLoop.current()
    if isinstance(io_loop.selector_loop, AddThreadSelectorEventLoop):
        self.skipTest('AddThreadSelectorEventLoop not supported')
    server, client = (yield self.make_iostream_pair())
    try:
        os.close(server.socket.fileno())
        with self.assertRaises(socket.error):
            server.read_bytes(1)
    finally:
        server.close()
        client.close()