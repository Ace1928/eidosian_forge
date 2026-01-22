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
def test_future_close_callback(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    closed = [False]
    cond = Condition()

    def close_callback():
        closed[0] = True
        cond.notify()
    rs.set_close_callback(close_callback)
    try:
        ws.write(b'a')
        res = (yield rs.read_bytes(1))
        self.assertEqual(res, b'a')
        self.assertFalse(closed[0])
        ws.close()
        yield cond.wait()
        self.assertTrue(closed[0])
    finally:
        rs.close()
        ws.close()