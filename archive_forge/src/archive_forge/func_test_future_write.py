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
def test_future_write(self):
    """
        Test that write() Futures are never orphaned.
        """
    m, n = (5000, 1000)
    nproducers = 10
    total_bytes = m * n * nproducers
    server, client = (yield self.make_iostream_pair(max_buffer_size=total_bytes))

    @gen.coroutine
    def produce():
        data = b'x' * m
        for i in range(n):
            yield server.write(data)

    @gen.coroutine
    def consume():
        nread = 0
        while nread < total_bytes:
            res = (yield client.read_bytes(m))
            nread += len(res)
    try:
        yield ([produce() for i in range(nproducers)] + [consume()])
    finally:
        server.close()
        client.close()