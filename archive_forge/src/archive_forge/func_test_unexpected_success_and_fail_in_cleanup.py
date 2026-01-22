import unittest
from unittest.test.support import LoggingResult
def test_unexpected_success_and_fail_in_cleanup(self):

    class Foo(unittest.TestCase):

        @unittest.expectedFailure
        def test_die(self):
            pass

        def tearDown(self):
            self.fail('bad tearDown')
    events = []
    result = LoggingResult(events)
    test = Foo('test_die')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addFailure', 'stopTest'])
    self.assertEqual(len(result.failures), 1)
    self.assertIn('AssertionError: bad tearDown', result.failures[0][1])
    self.assertFalse(result.expectedFailures)
    self.assertFalse(result.unexpectedSuccesses)
    self.assertFalse(result.wasSuccessful())