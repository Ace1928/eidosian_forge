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
def test_asyncio_run(self):
    for i in range(10):
        asyncio.run(self.dummy_tornado_coroutine())
    self.assert_no_thread_leak()