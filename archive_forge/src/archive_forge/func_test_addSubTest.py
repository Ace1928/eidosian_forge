import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def test_addSubTest(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            nonlocal subtest
            with self.subTest(foo=1):
                subtest = self._subtest
                try:
                    1 / 0
                except ZeroDivisionError:
                    exc_info_tuple = sys.exc_info()
                result.addSubTest(test, subtest, exc_info_tuple)
                self.fail('some recognizable failure')
    subtest = None
    test = Foo('test_1')
    result = unittest.TestResult()
    test.run(result)
    self.assertFalse(result.wasSuccessful())
    self.assertEqual(len(result.errors), 1)
    self.assertEqual(len(result.failures), 1)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(result.shouldStop, False)
    test_case, formatted_exc = result.errors[0]
    self.assertIs(test_case, subtest)
    self.assertIn('ZeroDivisionError', formatted_exc)
    test_case, formatted_exc = result.failures[0]
    self.assertIs(test_case, subtest)
    self.assertIn('some recognizable failure', formatted_exc)