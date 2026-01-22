import asyncio
import contextvars
import unittest
from test import support
def test_exception_in_tear_down(self):

    class Test(unittest.IsolatedAsyncioTestCase):

        async def asyncSetUp(self):
            events.append('asyncSetUp')

        async def test_func(self):
            events.append('test')
            self.addAsyncCleanup(self.on_cleanup)

        async def asyncTearDown(self):
            events.append('asyncTearDown')
            raise MyException()

        async def on_cleanup(self):
            events.append('cleanup')
    events = []
    test = Test('test_func')
    result = test.run()
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown', 'cleanup'])
    self.assertIs(result.errors[0][0], test)
    self.assertIn('MyException', result.errors[0][1])
    events = []
    test = Test('test_func')
    self.addCleanup(test._tearDownAsyncioRunner)
    try:
        test.debug()
    except MyException:
        pass
    else:
        self.fail('Expected a MyException exception')
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown'])
    test.doCleanups()
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown', 'cleanup'])