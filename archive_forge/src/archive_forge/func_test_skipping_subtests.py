import unittest
from unittest.test.support import LoggingResult
def test_skipping_subtests(self):

    class Foo(unittest.TestCase):

        def defaultTestResult(self):
            return LoggingResult(events)

        def test_skip_me(self):
            with self.subTest(a=1):
                with self.subTest(b=2):
                    self.skipTest('skip 1')
                self.skipTest('skip 2')
            self.skipTest('skip 3')
    events = []
    result = LoggingResult(events)
    test = Foo('test_skip_me')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'addSkip', 'addSkip', 'stopTest'])
    self.assertEqual(len(result.skipped), 3)
    subtest, msg = result.skipped[0]
    self.assertEqual(msg, 'skip 1')
    self.assertIsInstance(subtest, unittest.TestCase)
    self.assertIsNot(subtest, test)
    subtest, msg = result.skipped[1]
    self.assertEqual(msg, 'skip 2')
    self.assertIsInstance(subtest, unittest.TestCase)
    self.assertIsNot(subtest, test)
    self.assertEqual(result.skipped[2], (test, 'skip 3'))
    events = []
    result = test.run()
    self.assertEqual(events, ['startTestRun', 'startTest', 'addSkip', 'addSkip', 'addSkip', 'stopTest', 'stopTestRun'])
    self.assertEqual([msg for subtest, msg in result.skipped], ['skip 1', 'skip 2', 'skip 3'])