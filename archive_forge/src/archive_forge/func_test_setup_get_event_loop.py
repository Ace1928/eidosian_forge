import asyncio
import contextvars
import unittest
from test import support
def test_setup_get_event_loop(self):
    asyncio.set_event_loop(None)

    class TestCase1(unittest.IsolatedAsyncioTestCase):

        def setUp(self):
            asyncio.get_event_loop_policy().get_event_loop()

        async def test_demo1(self):
            pass
    test = TestCase1('test_demo1')
    result = test.run()
    self.assertTrue(result.wasSuccessful())