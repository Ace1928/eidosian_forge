import unittest
from unittest.test.support import LoggingResult
def test_skip_in_setup(self):

    class Foo(unittest.TestCase):

        def setUp(self):
            self.skipTest('skip')

        def test_skip_me(self):
            self.fail("shouldn't come here")
    events = []
    result = LoggingResult(events)
    test = Foo('test_skip_me')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertEqual(result.skipped, [(test, 'skip')])