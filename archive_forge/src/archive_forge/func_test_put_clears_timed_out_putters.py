import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_put_clears_timed_out_putters(self):
    q = queues.Queue(1)
    putters = [q.put(i, timedelta(seconds=0.01)) for i in range(10)]
    put = q.put(10)
    self.assertEqual(10, len(q._putters))
    yield gen.sleep(0.02)
    self.assertEqual(10, len(q._putters))
    self.assertFalse(put.done())
    q.put(11)
    self.assertEqual(2, len(q._putters))
    for putter in putters[1:]:
        self.assertRaises(TimeoutError, putter.result)