from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
def test_subsequent_wait_calls(self):
    """
        This test makes sure that a second call to wait()
        clears the first timeout.
        """
    self.io_loop.add_timeout(self.io_loop.time() + 0.0, self.stop)
    self.wait(timeout=0.1)
    self.io_loop.add_timeout(self.io_loop.time() + 0.2, self.stop)
    self.wait(timeout=0.4)