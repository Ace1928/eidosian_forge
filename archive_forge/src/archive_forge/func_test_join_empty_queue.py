import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_join_empty_queue(self):
    q = self.queue_class()
    yield q.join()
    yield q.join()