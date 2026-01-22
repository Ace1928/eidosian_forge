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
def test_propagate_to_first_yield_with_native_async_function(self):
    x = 10

    async def native_async_function():
        self.assertEqual(ctx_var.get(), x)
    ctx_var.set(x)
    yield native_async_function()