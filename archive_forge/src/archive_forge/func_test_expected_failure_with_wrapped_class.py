import unittest
from unittest.test.support import LoggingResult
def test_expected_failure_with_wrapped_class(self):

    @unittest.expectedFailure
    class Foo(unittest.TestCase):

        def test_1(self):
            self.assertTrue(False)
    events = []
    result = LoggingResult(events)
    test = Foo('test_1')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addExpectedFailure', 'stopTest'])
    self.assertFalse(result.failures)
    self.assertEqual(result.expectedFailures[0][0], test)
    self.assertFalse(result.unexpectedSuccesses)
    self.assertTrue(result.wasSuccessful())