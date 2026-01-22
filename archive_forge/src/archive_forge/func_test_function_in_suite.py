import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_function_in_suite(self):

    def f(_):
        pass
    suite = unittest.TestSuite()
    suite.addTest(f)
    suite.run(unittest.TestResult())