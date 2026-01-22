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
@skipOnTravis
def test_add_callback_wakeup(self):

    def callback():
        self.called = True
        self.stop()

    def schedule_callback():
        self.called = False
        self.io_loop.add_callback(callback)
        self.start_time = time.time()
    self.io_loop.add_timeout(self.io_loop.time(), schedule_callback)
    self.wait()
    self.assertAlmostEqual(time.time(), self.start_time, places=2)
    self.assertTrue(self.called)