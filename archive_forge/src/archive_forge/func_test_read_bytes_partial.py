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
@gen_test
def test_read_bytes_partial(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    try:
        fut = rs.read_bytes(50, partial=True)
        ws.write(b'hello')
        data = (yield fut)
        self.assertEqual(data, b'hello')
        fut = rs.read_bytes(3, partial=True)
        ws.write(b'world')
        data = (yield fut)
        self.assertEqual(data, b'wor')
        data = (yield rs.read_bytes(0, partial=True))
        self.assertEqual(data, b'')
    finally:
        ws.close()
        rs.close()