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
def test_remove_handler_from_handler(self):
    client, server = socket.socketpair()
    try:
        client.send(b'abc')
        server.send(b'abc')
        chunks = []

        def handle_read(fd, events):
            chunks.append(fd.recv(1024))
            if fd is client:
                self.io_loop.remove_handler(server)
            else:
                self.io_loop.remove_handler(client)
        self.io_loop.add_handler(client, handle_read, self.io_loop.READ)
        self.io_loop.add_handler(server, handle_read, self.io_loop.READ)
        self.io_loop.call_later(0.1, self.stop)
        self.wait()
        self.assertEqual(chunks, [b'abc'])
    finally:
        client.close()
        server.close()