import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_invalid_testmethod(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['testcase_1.testfoo'], m)
    error, test = self.check_deferred_error(loader, list(suite)[0])
    expected = "type object 'MyTestCase' has no attribute 'testfoo'"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, test.testfoo)