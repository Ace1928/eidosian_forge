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
def test_add_callback_wakeup_other_thread(self):

    def target():
        time.sleep(0.01)
        self.stop_time = time.time()
        self.io_loop.add_callback(self.stop)
    thread = threading.Thread(target=target)
    self.io_loop.add_callback(thread.start)
    self.wait()
    delta = time.time() - self.stop_time
    self.assertLess(delta, 0.1)
    thread.join()