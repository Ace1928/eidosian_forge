import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_init__tests_from_any_iterable(self):

    def tests():
        yield unittest.FunctionTestCase(lambda: None)
        yield unittest.FunctionTestCase(lambda: None)
    suite_1 = unittest.TestSuite(tests())
    self.assertEqual(suite_1.countTestCases(), 2)
    suite_2 = unittest.TestSuite(suite_1)
    self.assertEqual(suite_2.countTestCases(), 2)
    suite_3 = unittest.TestSuite(set(suite_1))
    self.assertEqual(suite_3.countTestCases(), 2)
    suite_1.run(unittest.TestResult())
    self.assertEqual(suite_1.countTestCases(), 2)
    suite_2.run(unittest.TestResult())
    self.assertEqual(suite_2.countTestCases(), 2)
    suite_3.run(unittest.TestResult())
    self.assertEqual(suite_3.countTestCases(), 2)