import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__no_TestCase_tests(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):
        pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [loader.suiteClass()])