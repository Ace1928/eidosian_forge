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
def test_no_timeout_environment_variable(self):

    @gen_test(timeout=0.01)
    def test_short_timeout(self):
        yield gen.sleep(1)
    with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
        with self.assertRaises(ioloop.TimeoutError):
            test_short_timeout(self)
    self.finished = True