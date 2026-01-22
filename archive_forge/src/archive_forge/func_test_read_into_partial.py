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
def test_read_into_partial(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    try:
        buf = bytearray(10)
        fut = rs.read_into(buf, partial=True)
        ws.write(b'hello')
        data = (yield fut)
        self.assertFalse(rs.reading())
        self.assertEqual(data, 5)
        self.assertEqual(bytes(buf), b'hello\x00\x00\x00\x00\x00')
        ws.write(b'world!1234567890')
        data = (yield rs.read_into(buf, partial=True))
        self.assertEqual(data, 10)
        self.assertEqual(bytes(buf), b'world!1234')
        data = (yield rs.read_into(buf, partial=True))
        self.assertEqual(data, 6)
        self.assertEqual(bytes(buf), b'5678901234')
    finally:
        ws.close()
        rs.close()