import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_acquire_timeout_preempted(self):
    sem = locks.Semaphore(1)
    yield sem.acquire()
    self.io_loop.call_later(0.01, sem.release)
    acquire = sem.acquire(timedelta(seconds=0.02))
    yield gen.sleep(0.03)
    yield acquire