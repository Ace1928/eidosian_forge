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
def test_timeout_environment_variable(self):

    @gen_test(timeout=0.5)
    def test_long_timeout(self):
        yield gen.sleep(0.25)
    with set_environ('ASYNC_TEST_TIMEOUT', '0.1'):
        test_long_timeout(self)
    self.finished = True