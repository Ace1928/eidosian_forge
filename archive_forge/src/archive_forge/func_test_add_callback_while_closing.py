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
def test_add_callback_while_closing(self):
    closing = threading.Event()

    def target():
        other_ioloop.add_callback(other_ioloop.stop)
        other_ioloop.start()
        closing.set()
        other_ioloop.close(all_fds=True)
    other_ioloop = IOLoop()
    thread = threading.Thread(target=target)
    thread.start()
    closing.wait()
    for i in range(1000):
        other_ioloop.add_callback(lambda: None)