import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_repr_and_str(self):
    q = queues.Queue(maxsize=1)
    self.assertIn(hex(id(q)), repr(q))
    self.assertNotIn(hex(id(q)), str(q))
    q.get()
    for q_str in (repr(q), str(q)):
        self.assertTrue(q_str.startswith('<Queue'))
        self.assertIn('maxsize=1', q_str)
        self.assertIn('getters[1]', q_str)
        self.assertNotIn('putters', q_str)
        self.assertNotIn('tasks', q_str)
    q.put(None)
    q.put(None)
    q.put(None)
    for q_str in (repr(q), str(q)):
        self.assertNotIn('getters', q_str)
        self.assertIn('putters[1]', q_str)
        self.assertIn('tasks=2', q_str)