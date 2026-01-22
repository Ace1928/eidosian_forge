import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_remove_test_at_index(self):
    if not unittest.BaseTestSuite._cleanup:
        raise unittest.SkipTest('Suite cleanup is disabled')
    suite = unittest.TestSuite()
    suite._tests = [1, 2, 3]
    suite._removeTestAtIndex(1)
    self.assertEqual([1, None, 3], suite._tests)