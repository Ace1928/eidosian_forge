import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_blocking_get_wait(self):
    q = queues.Queue()
    q.put(0)
    self.io_loop.call_later(0.01, q.put_nowait, 1)
    self.io_loop.call_later(0.02, q.put_nowait, 2)
    self.assertEqual(0, (yield q.get(timeout=timedelta(seconds=1))))
    self.assertEqual(1, (yield q.get(timeout=timedelta(seconds=1))))