import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_context_manager_async_await(self):
    sem = locks.Semaphore()

    async def f():
        async with sem as yielded:
            self.assertTrue(yielded is None)
    yield f()
    self.assertTrue(asyncio.ensure_future(sem.acquire()).done())