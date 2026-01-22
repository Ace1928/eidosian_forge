import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
from tornado.testing import AsyncTestCase, gen_test
@gen_test
def test_asyncio_yield_from(self):

    @gen.coroutine
    def f():
        event_loop = asyncio.get_event_loop()
        x = (yield from event_loop.run_in_executor(None, lambda: 42))
        return x
    result = (yield f())
    self.assertEqual(result, 42)