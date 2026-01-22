import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_yield_sem(self):
    with self.assertRaises(gen.BadYieldError):
        with (yield locks.Semaphore()):
            pass