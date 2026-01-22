import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_nonblocking_get_exception(self):
    q = queues.Queue()
    self.assertRaises(queues.QueueEmpty, q.get_nowait)