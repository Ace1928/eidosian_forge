import unittest
from unittest.test.support import LoggingResult
def test_unexpected_success_subtests(self):

    class Foo(unittest.TestCase):

        @unittest.expectedFailure
        def test_die(self):
            with self.subTest():
                pass
            with self.subTest():
                pass
    events = []
    result = LoggingResult(events)
    test = Foo('test_die')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSubTestSuccess', 'addSubTestSuccess', 'addUnexpectedSuccess', 'stopTest'])
    self.assertFalse(result.failures)
    self.assertFalse(result.expectedFailures)
    self.assertEqual(result.unexpectedSuccesses, [test])
    self.assertFalse(result.wasSuccessful())