import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTest__casesuiteclass(self):
    suite = unittest.TestSuite()
    self.assertRaises(TypeError, suite.addTest, Test_TestSuite)
    self.assertRaises(TypeError, suite.addTest, unittest.TestSuite)