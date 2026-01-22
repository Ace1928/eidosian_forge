import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_event_timeout(self):
    e = locks.Event()
    with self.assertRaises(TimeoutError):
        yield e.wait(timedelta(seconds=0.01))
    self.io_loop.add_timeout(timedelta(seconds=0.01), e.set)
    yield e.wait(timedelta(seconds=1))