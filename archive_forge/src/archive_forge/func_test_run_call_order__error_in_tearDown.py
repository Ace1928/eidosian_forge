import unittest
from unittest.test.support import LoggingResult
def test_run_call_order__error_in_tearDown(self):
    events = []
    result = LoggingResult(events)

    def setUp():
        events.append('setUp')

    def test():
        events.append('test')

    def tearDown():
        events.append('tearDown')
        raise RuntimeError('raised by tearDown')
    expected = ['startTest', 'setUp', 'test', 'tearDown', 'addError', 'stopTest']
    unittest.FunctionTestCase(test, setUp, tearDown).run(result)
    self.assertEqual(events, expected)