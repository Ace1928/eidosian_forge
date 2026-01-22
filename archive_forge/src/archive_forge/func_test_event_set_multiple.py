import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_event_set_multiple(self):
    e = locks.Event()
    e.set()
    e.set()
    self.assertTrue(e.is_set())