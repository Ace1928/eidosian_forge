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
def test_future_delayed_close_callback(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair())
    try:
        ws.write(b'12')
        chunks = []
        chunks.append((yield rs.read_bytes(1)))
        ws.close()
        chunks.append((yield rs.read_bytes(1)))
        self.assertEqual(chunks, [b'1', b'2'])
    finally:
        ws.close()
        rs.close()