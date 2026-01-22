import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferDoCleanups(self):
    with captured_stdout() as stdout:
        result = unittest.TestResult()
    result.buffer = True

    class Foo(unittest.TestCase):

        def setUp(self):
            print('set up')
            self.addCleanup(bad_cleanup1)
            self.addCleanup(bad_cleanup2)

        def test_foo(self):
            pass
    suite = unittest.TestSuite([Foo('test_foo')])
    suite(result)
    expected_out = '\nStdout:\nset up\ndo cleanup2\ndo cleanup1\n'
    self.assertEqual(stdout.getvalue(), expected_out)
    self.assertEqual(len(result.errors), 2)
    description = f'test_foo ({strclass(Foo)}.test_foo)'
    test_case, formatted_exc = result.errors[0]
    self.assertEqual(str(test_case), description)
    self.assertIn('ValueError: bad cleanup2', formatted_exc)
    self.assertNotIn('TypeError', formatted_exc)
    self.assertIn('\nStdout:\nset up\ndo cleanup2\n', formatted_exc)
    self.assertNotIn('\ndo cleanup1\n', formatted_exc)
    test_case, formatted_exc = result.errors[1]
    self.assertEqual(str(test_case), description)
    self.assertIn('TypeError: bad cleanup1', formatted_exc)
    self.assertNotIn('ValueError', formatted_exc)
    self.assertIn(expected_out, formatted_exc)