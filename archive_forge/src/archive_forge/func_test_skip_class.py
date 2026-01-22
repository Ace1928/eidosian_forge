import unittest
from unittest.test.support import LoggingResult
def test_skip_class(self):

    @unittest.skip('testing')
    class Foo(unittest.TestCase):

        def defaultTestResult(self):
            return LoggingResult(events)

        def test_1(self):
            record.append(1)
    events = []
    record = []
    result = LoggingResult(events)
    test = Foo('test_1')
    suite = unittest.TestSuite([test])
    self.assertIs(suite.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertEqual(record, [])
    events = []
    result = test.run()
    self.assertEqual(events, ['startTestRun', 'startTest', 'addSkip', 'stopTest', 'stopTestRun'])
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertEqual(record, [])