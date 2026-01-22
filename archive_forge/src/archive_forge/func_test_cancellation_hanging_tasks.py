import asyncio
import contextvars
import unittest
from test import support
def test_cancellation_hanging_tasks(self):
    cancelled = False

    class Test(unittest.IsolatedAsyncioTestCase):

        async def test_leaking_task(self):

            async def coro():
                nonlocal cancelled
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    cancelled = True
                    raise
            asyncio.create_task(coro())
    test = Test('test_leaking_task')
    output = test.run()
    self.assertTrue(cancelled)