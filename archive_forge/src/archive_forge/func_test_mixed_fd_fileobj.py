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
def test_mixed_fd_fileobj(self):
    server_sock, port = bind_unused_port()

    def f(fd, events):
        pass
    self.io_loop.add_handler(server_sock, f, IOLoop.READ)
    with self.assertRaises(Exception):
        self.io_loop.add_handler(server_sock.fileno(), f, IOLoop.READ)
    self.io_loop.remove_handler(server_sock.fileno())
    server_sock.close()