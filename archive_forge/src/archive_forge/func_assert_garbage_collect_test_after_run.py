import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def assert_garbage_collect_test_after_run(self, TestSuiteClass):
    if not unittest.BaseTestSuite._cleanup:
        raise unittest.SkipTest('Suite cleanup is disabled')

    class Foo(unittest.TestCase):

        def test_nothing(self):
            pass
    test = Foo('test_nothing')
    wref = weakref.ref(test)
    suite = TestSuiteClass([wref()])
    suite.run(unittest.TestResult())
    del test
    gc.collect()
    self.assertEqual(suite._tests, [None])
    self.assertIsNone(wref())