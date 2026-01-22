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
def check_peek(self, buf, expected):
    size = 1
    while size < 2 * len(expected):
        got = self.to_bytes(buf.peek(size))
        self.assertTrue(got)
        self.assertLessEqual(len(got), size)
        self.assertTrue(expected.startswith(got), (expected, got))
        size = (size * 3 + 1) // 2