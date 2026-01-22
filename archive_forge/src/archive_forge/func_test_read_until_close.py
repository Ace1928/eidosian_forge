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
def test_read_until_close(self: typing.Any):
    stream = self._make_client_iostream()
    yield stream.connect(('127.0.0.1', self.get_http_port()))
    stream.write(b'GET / HTTP/1.0\r\n\r\n')
    data = (yield stream.read_until_close())
    self.assertTrue(data.startswith(b'HTTP/1.1 200'))
    self.assertTrue(data.endswith(b'Hello'))