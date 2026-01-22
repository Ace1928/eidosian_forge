import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
def test_exception_logging_native_coro(self):
    """The IOLoop examines exceptions from awaitables and logs them."""

    async def callback():
        self.io_loop.add_callback(self.io_loop.add_callback, self.stop)
        1 / 0
    self.io_loop.add_callback(callback)
    with ExpectLog(app_log, 'Exception in callback'):
        self.wait()