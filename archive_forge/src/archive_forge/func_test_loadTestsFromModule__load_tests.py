import functools
import sys
import types
import warnings
import unittest
@warningregistry
def test_loadTestsFromModule__load_tests(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    load_tests_args = []

    def load_tests(loader, tests, pattern):
        self.assertIsInstance(tests, unittest.TestSuite)
        load_tests_args.extend((loader, tests, pattern))
        return tests
    m.load_tests = load_tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, unittest.TestSuite)
    self.assertEqual(load_tests_args, [loader, suite, None])
    load_tests_args = []
    with warnings.catch_warnings(record=False):
        warnings.simplefilter('ignore')
        suite = loader.loadTestsFromModule(m, use_load_tests=False)
    self.assertEqual(load_tests_args, [loader, suite, None])