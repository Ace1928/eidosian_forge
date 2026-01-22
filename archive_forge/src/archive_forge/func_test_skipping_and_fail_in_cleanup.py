import unittest
from unittest.test.support import LoggingResult
def test_skipping_and_fail_in_cleanup(self):

    class Foo(unittest.TestCase):

        def test_skip_me(self):
            self.skipTest('skip')

        def tearDown(self):
            self.fail('fail')
    events = []
    result = LoggingResult(events)
    test = Foo('test_skip_me')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'addFailure', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'skip')])