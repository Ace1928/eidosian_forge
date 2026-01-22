import unittest
from unittest.test.support import LoggingResult
def test_countTestCases(self):
    test = unittest.FunctionTestCase(lambda: None)
    self.assertEqual(test.countTestCases(), 1)