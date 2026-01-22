import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_mutter_callsite_2(self):
    """mutter_callsite can capture 2 levels of stack frame."""
    mutter_callsite(2, 'foo %s', 'a string')
    log = self.get_log()
    self.assertLogContainsLine(log, 'foo a string\nCalled from:\n')
    self.assertContainsRe(log, 'test_trace.py", line \\d+, in test_mutter_callsite_2\n')
    self.assertEndsWith(log, ' "a string")\n')