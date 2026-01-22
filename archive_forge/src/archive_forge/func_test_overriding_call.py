import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_overriding_call(self):

    class MySuite(unittest.TestSuite):
        called = False

        def __call__(self, *args, **kw):
            self.called = True
            unittest.TestSuite.__call__(self, *args, **kw)
    suite = MySuite()
    result = unittest.TestResult()
    wrapper = unittest.TestSuite()
    wrapper.addTest(suite)
    wrapper(result)
    self.assertTrue(suite.called)
    self.assertFalse(result._testRunEntered)