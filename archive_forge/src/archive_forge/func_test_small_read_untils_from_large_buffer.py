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
def test_small_read_untils_from_large_buffer(self: typing.Any):
    rs, ws = (yield self.make_iostream_pair(max_buffer_size=10 * 1024))
    try:
        ws.write((b'a' * 1023 + b'\n') * 100)
        for i in range(100):
            data = (yield rs.read_until(b'\n', max_bytes=4096))
            self.assertEqual(data, b'a' * 1023 + b'\n')
    finally:
        ws.close()
        rs.close()