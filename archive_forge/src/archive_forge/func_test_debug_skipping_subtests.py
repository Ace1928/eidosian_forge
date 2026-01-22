import unittest
from unittest.test.support import LoggingResult
def test_debug_skipping_subtests(self):

    class Foo(unittest.TestCase):

        def setUp(self):
            events.append('setUp')

        def tearDown(self):
            events.append('tearDown')

        def test(self):
            with self.subTest(a=1):
                events.append('subtest')
                self.skipTest('skip subtest')
                events.append('end subtest')
            events.append('end test')
    events = []
    result = LoggingResult(events)
    test = Foo('test')
    with self.assertRaises(unittest.SkipTest) as cm:
        test.debug()
    self.assertIn('skip subtest', str(cm.exception))
    self.assertEqual(events, ['setUp', 'subtest'])