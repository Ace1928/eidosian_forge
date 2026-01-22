import unittest
from unittest.test.support import LoggingResult
def test_skipping(self):

    class Foo(unittest.TestCase):

        def defaultTestResult(self):
            return LoggingResult(events)

        def test_skip_me(self):
            self.skipTest('skip')
    events = []
    result = LoggingResult(events)
    test = Foo('test_skip_me')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'skip')])
    events = []
    result = test.run()
    self.assertEqual(events, ['startTestRun', 'startTest', 'addSkip', 'stopTest', 'stopTestRun'])
    self.assertEqual(result.skipped, [(test, 'skip')])
    self.assertEqual(result.testsRun, 1)

    class Foo(unittest.TestCase):

        def defaultTestResult(self):
            return LoggingResult(events)

        def setUp(self):
            self.skipTest('testing')

        def test_nothing(self):
            pass
    events = []
    result = LoggingResult(events)
    test = Foo('test_nothing')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertEqual(result.testsRun, 1)
    events = []
    result = test.run()
    self.assertEqual(events, ['startTestRun', 'startTest', 'addSkip', 'stopTest', 'stopTestRun'])
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertEqual(result.testsRun, 1)