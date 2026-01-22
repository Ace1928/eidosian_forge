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
def test_periodic_plain(self):
    count = 0

    def callback() -> None:
        nonlocal count
        count += 1
        if count == 3:
            self.stop()
    pc = PeriodicCallback(callback, 10)
    pc.start()
    self.wait()
    pc.stop()
    self.assertEqual(count, 3)