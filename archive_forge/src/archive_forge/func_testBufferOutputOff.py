import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferOutputOff(self):
    real_out = self._real_out
    real_err = self._real_err
    result = unittest.TestResult()
    self.assertFalse(result.buffer)
    self.assertIs(real_out, sys.stdout)
    self.assertIs(real_err, sys.stderr)
    result.startTest(self)
    self.assertIs(real_out, sys.stdout)
    self.assertIs(real_err, sys.stderr)