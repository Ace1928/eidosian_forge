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
def test_jitter(self):
    random_times = [0.5, 1, 0, 0.75]
    expected = [1010, 1022.5, 1030, 1041.25]
    call_durations = [0] * len(random_times)
    pc = PeriodicCallback(self.dummy, 10000, jitter=0.5)

    def mock_random():
        return random_times.pop(0)
    with mock.patch('random.random', mock_random):
        self.assertEqual(self.simulate_calls(pc, call_durations), expected)