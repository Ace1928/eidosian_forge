import unittest
from unittest.test.support import LoggingResult
def test_run_call_order__failure_in_test(self):
    events = []
    result = LoggingResult(events)

    def setUp():
        events.append('setUp')

    def test():
        events.append('test')
        self.fail('raised by test')

    def tearDown():
        events.append('tearDown')
    expected = ['startTest', 'setUp', 'test', 'addFailure', 'tearDown', 'stopTest']
    unittest.FunctionTestCase(test, setUp, tearDown).run(result)
    self.assertEqual(events, expected)