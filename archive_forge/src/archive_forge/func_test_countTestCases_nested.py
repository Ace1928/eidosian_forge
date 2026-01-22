import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_countTestCases_nested(self):

    class Test1(unittest.TestCase):

        def test1(self):
            pass

        def test2(self):
            pass
    test2 = unittest.FunctionTestCase(lambda: None)
    test3 = unittest.FunctionTestCase(lambda: None)
    child = unittest.TestSuite((Test1('test2'), test2))
    parent = unittest.TestSuite((test3, child, Test1('test1')))
    self.assertEqual(parent.countTestCases(), 4)
    parent.run(unittest.TestResult())
    self.assertEqual(parent.countTestCases(), 4)
    self.assertEqual(child.countTestCases(), 2)