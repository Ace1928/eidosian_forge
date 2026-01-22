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
def iostream_pair(self, **kwargs):
    """Like make_iostream_pair, but called by ``async with``.

        In py37 this becomes simpler with contextlib.asynccontextmanager.
        """

    class IOStreamPairContext:

        def __init__(self, test, kwargs):
            self.test = test
            self.kwargs = kwargs

        async def __aenter__(self):
            self.pair = await self.test.make_iostream_pair(**self.kwargs)
            return self.pair

        async def __aexit__(self, typ, value, tb):
            for s in self.pair:
                s.close()
    return IOStreamPairContext(self, kwargs)