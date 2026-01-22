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
def check_append_all_then_skip_all(self, buf, objs, input_type):
    self.assertEqual(len(buf), 0)
    expected = b''
    for o in objs:
        expected += o
        buf.append(input_type(o))
        self.assertEqual(len(buf), len(expected))
        self.check_peek(buf, expected)
    while expected:
        n = self.random.randrange(1, len(expected) + 1)
        expected = expected[n:]
        buf.advance(n)
        self.assertEqual(len(buf), len(expected))
        self.check_peek(buf, expected)
    self.assertEqual(len(buf), 0)