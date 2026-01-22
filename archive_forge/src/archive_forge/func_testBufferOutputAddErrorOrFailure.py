import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferOutputAddErrorOrFailure(self):
    unittest.result.traceback = MockTraceback
    self.addCleanup(restore_traceback)
    for message_attr, add_attr, include_error in [('errors', 'addError', True), ('failures', 'addFailure', False), ('errors', 'addError', True), ('failures', 'addFailure', False)]:
        result = self.getStartedResult()
        buffered_out = sys.stdout
        buffered_err = sys.stderr
        result._original_stdout = io.StringIO()
        result._original_stderr = io.StringIO()
        print('foo', file=sys.stdout)
        if include_error:
            print('bar', file=sys.stderr)
        addFunction = getattr(result, add_attr)
        addFunction(self, (None, None, None))
        result.stopTest(self)
        result_list = getattr(result, message_attr)
        self.assertEqual(len(result_list), 1)
        test, message = result_list[0]
        expectedOutMessage = textwrap.dedent('\n                Stdout:\n                foo\n            ')
        expectedErrMessage = ''
        if include_error:
            expectedErrMessage = textwrap.dedent('\n                Stderr:\n                bar\n            ')
        expectedFullMessage = 'A traceback%s%s' % (expectedOutMessage, expectedErrMessage)
        self.assertIs(test, self)
        self.assertEqual(result._original_stdout.getvalue(), expectedOutMessage)
        self.assertEqual(result._original_stderr.getvalue(), expectedErrMessage)
        self.assertMultiLineEqual(message, expectedFullMessage)