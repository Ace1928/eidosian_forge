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
def test_timeout_with_arguments(self):
    results = []
    self.io_loop.add_timeout(self.io_loop.time(), results.append, 1)
    self.io_loop.add_timeout(datetime.timedelta(seconds=0), results.append, 2)
    self.io_loop.call_at(self.io_loop.time(), results.append, 3)
    self.io_loop.call_later(0, results.append, 4)
    self.io_loop.call_later(0, self.stop)
    self.wait()
    self.assertEqual(sorted(results), [1, 2, 3, 4])