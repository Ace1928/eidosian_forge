import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_TestCase_subclass(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['testcase_1'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    expected = loader.suiteClass([MyTestCase('test')])
    self.assertEqual(list(suite), [expected])