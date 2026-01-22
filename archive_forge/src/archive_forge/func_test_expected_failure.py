import unittest
from unittest.test.support import LoggingResult
def test_expected_failure(self):

    class Foo(unittest.TestCase):

        @unittest.expectedFailure
        def test_die(self):
            self.fail('help me!')
    events = []
    result = LoggingResult(events)
    test = Foo('test_die')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addExpectedFailure', 'stopTest'])
    self.assertFalse(result.failures)
    self.assertEqual(result.expectedFailures[0][0], test)
    self.assertFalse(result.unexpectedSuccesses)
    self.assertTrue(result.wasSuccessful())