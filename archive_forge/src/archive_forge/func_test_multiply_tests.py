from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def test_multiply_tests(self):
    loader = TestLoader()
    suite = loader.suiteClass()
    multiply_tests(self, vary_by_color(), suite)
    self.assertEqual(['blue', 'green', 'red'], get_generated_test_attributes(suite, 'color'))