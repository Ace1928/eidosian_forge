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
def test_add_callback_return_sequence(self):
    self.calls = 0
    loop = self.io_loop
    test = self
    old_add_callback = loop.add_callback

    def add_callback(self, callback, *args, **kwargs):
        test.calls += 1
        old_add_callback(callback, *args, **kwargs)
    loop.add_callback = types.MethodType(add_callback, loop)
    loop.add_callback(lambda: {})
    loop.add_callback(lambda: [])
    loop.add_timeout(datetime.timedelta(milliseconds=50), loop.stop)
    loop.start()
    self.assertLess(self.calls, 10)