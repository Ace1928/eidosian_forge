import asyncio
import contextvars
import unittest
from test import support
def test_enterAsyncContext(self):
    events = []

    class Test(unittest.IsolatedAsyncioTestCase):

        async def test_func(slf):
            slf.addAsyncCleanup(events.append, 'cleanup1')
            cm = TestCM(events, 42)
            self.assertEqual(await slf.enterAsyncContext(cm), 42)
            slf.addAsyncCleanup(events.append, 'cleanup2')
            events.append('test')
    test = Test('test_func')
    output = test.run()
    self.assertTrue(output.wasSuccessful(), output)
    self.assertEqual(events, ['enter', 'test', 'cleanup2', 'exit', 'cleanup1'])