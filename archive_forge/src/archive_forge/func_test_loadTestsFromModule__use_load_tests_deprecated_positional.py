import functools
import sys
import types
import warnings
import unittest
@warningregistry
def test_loadTestsFromModule__use_load_tests_deprecated_positional(self):
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        suite = loader.loadTestsFromModule(m, False)
    self.assertIsInstance(suite, unittest.TestSuite)
    self.assertEqual(load_tests_args, [loader, suite, None])
    self.assertIs(w[-1].category, DeprecationWarning)
    self.assertEqual(str(w[-1].message), 'use_load_tests is deprecated and ignored')