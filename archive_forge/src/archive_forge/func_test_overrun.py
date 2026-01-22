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
def test_overrun(self):
    call_durations = [9, 9, 10, 11, 20, 20, 35, 35, 0, 0, 0]
    expected = [1010, 1020, 1030, 1050, 1070, 1100, 1130, 1170, 1210, 1220, 1230]
    pc = PeriodicCallback(self.dummy, 10000)
    self.assertEqual(self.simulate_calls(pc, call_durations), expected)