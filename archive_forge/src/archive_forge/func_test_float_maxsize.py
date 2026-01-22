import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_float_maxsize(self):
    q = queues.Queue(maxsize=1.3)
    self.assertTrue(q.empty())
    self.assertFalse(q.full())
    q.put_nowait(0)
    q.put_nowait(1)
    self.assertFalse(q.empty())
    self.assertTrue(q.full())
    self.assertRaises(queues.QueueFull, q.put_nowait, 2)
    self.assertEqual(0, q.get_nowait())
    self.assertFalse(q.empty())
    self.assertFalse(q.full())
    yield q.put(2)
    put = q.put(3)
    self.assertFalse(put.done())
    self.assertEqual(1, (yield q.get()))
    yield put
    self.assertTrue(q.full())