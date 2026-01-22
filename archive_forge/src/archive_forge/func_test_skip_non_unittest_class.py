import unittest
from unittest.test.support import LoggingResult
def test_skip_non_unittest_class(self):

    @unittest.skip('testing')
    class Mixin:

        def test_1(self):
            record.append(1)

    class Foo(Mixin, unittest.TestCase):
        pass
    record = []
    result = unittest.TestResult()
    test = Foo('test_1')
    suite = unittest.TestSuite([test])
    self.assertIs(suite.run(result), result)
    self.assertEqual(result.skipped, [(test, 'testing')])
    self.assertEqual(record, [])