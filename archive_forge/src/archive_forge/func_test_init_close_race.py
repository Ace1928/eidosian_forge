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
@skipIfNonUnix
@gen_test
def test_init_close_race(self):

    def f():
        for i in range(10):
            loop = IOLoop(make_current=False)
            loop.close()
    yield gen.multi([self.io_loop.run_in_executor(None, f) for i in range(2)])