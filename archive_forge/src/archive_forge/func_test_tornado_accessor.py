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
def test_tornado_accessor(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
        self.executor.submit(lambda: asyncio.get_event_loop().close()).result()
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        self.assertIsInstance(self.executor.submit(IOLoop.current).result(), IOLoop)
        self.executor.submit(lambda: asyncio.get_event_loop().close()).result()