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
def test_explicit_asyncio_loop(self):
    asyncio_loop = asyncio.new_event_loop()
    loop = IOLoop(asyncio_loop=asyncio_loop, make_current=False)
    assert loop.asyncio_loop is asyncio_loop
    with self.assertRaises(RuntimeError):
        IOLoop(asyncio_loop=asyncio_loop, make_current=False)
    loop.close()