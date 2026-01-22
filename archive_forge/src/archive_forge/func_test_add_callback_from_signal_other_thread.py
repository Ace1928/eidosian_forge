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
def test_add_callback_from_signal_other_thread(self):
    other_ioloop = IOLoop()
    thread = threading.Thread(target=other_ioloop.start)
    thread.start()
    with ignore_deprecation():
        other_ioloop.add_callback_from_signal(other_ioloop.stop)
    thread.join()
    other_ioloop.close()