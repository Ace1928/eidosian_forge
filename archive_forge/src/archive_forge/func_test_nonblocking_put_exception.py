import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_nonblocking_put_exception(self):
    q = queues.Queue(1)
    q.put(0)
    self.assertRaises(queues.QueueFull, q.put_nowait, 1)