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
def test_is_coroutine_function(self):
    self.finished = True

    def f():
        yield gen.moment
    coro = gen.coroutine(f)
    self.assertFalse(gen.is_coroutine_function(f))
    self.assertTrue(gen.is_coroutine_function(coro))
    self.assertFalse(gen.is_coroutine_function(coro()))