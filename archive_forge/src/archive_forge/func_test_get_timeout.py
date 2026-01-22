import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_get_timeout(self):
    q = queues.Queue()
    get_timeout = q.get(timeout=timedelta(seconds=0.01))
    get = q.get()
    with self.assertRaises(TimeoutError):
        yield get_timeout
    q.put_nowait(0)
    self.assertEqual(0, (yield get))