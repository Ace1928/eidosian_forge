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
def test_tornado(self):
    for i in range(10):
        loop = IOLoop(make_current=False)
        loop.run_sync(self.dummy_tornado_coroutine)
        loop.close()
    self.assert_no_thread_leak()