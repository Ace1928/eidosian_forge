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
@gen_test
def test_set_default_executor(self):
    count = [0]

    class MyExecutor(futures.ThreadPoolExecutor):

        def submit(self, func, *args):
            count[0] += 1
            return super().submit(func, *args)
    event = threading.Event()

    def sync_func():
        event.set()
    executor = MyExecutor(1)
    loop = IOLoop.current()
    loop.set_default_executor(executor)
    yield loop.run_in_executor(None, sync_func)
    self.assertEqual(1, count[0])
    self.assertTrue(event.is_set())