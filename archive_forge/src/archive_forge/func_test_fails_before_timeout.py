import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
@gen_test
def test_fails_before_timeout(self):
    future = Future()
    self.io_loop.add_timeout(datetime.timedelta(seconds=0.1), lambda: future.set_exception(ZeroDivisionError()))
    with self.assertRaises(ZeroDivisionError):
        yield gen.with_timeout(datetime.timedelta(seconds=3600), future)