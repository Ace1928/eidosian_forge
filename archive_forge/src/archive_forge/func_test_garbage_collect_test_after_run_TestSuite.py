import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_garbage_collect_test_after_run_TestSuite(self):
    self.assert_garbage_collect_test_after_run(unittest.TestSuite)