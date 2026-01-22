import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testOldTestResult(self):

    class Test(unittest.TestCase):

        def testSkip(self):
            self.skipTest('foobar')

        @unittest.expectedFailure
        def testExpectedFail(self):
            raise TypeError

        @unittest.expectedFailure
        def testUnexpectedSuccess(self):
            pass
    for test_name, should_pass in (('testSkip', True), ('testExpectedFail', True), ('testUnexpectedSuccess', False)):
        test = Test(test_name)
        self.assertOldResultWarning(test, int(not should_pass))