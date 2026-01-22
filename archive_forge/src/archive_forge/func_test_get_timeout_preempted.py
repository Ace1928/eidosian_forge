import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_get_timeout_preempted(self):
    q = queues.Queue()
    get = q.get(timeout=timedelta(seconds=0.01))
    q.put(0)
    yield gen.sleep(0.02)
    self.assertEqual(0, (yield get))