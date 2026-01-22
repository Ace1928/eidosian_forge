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
def test_mutter_callsite_1(self):
    """mutter_callsite can capture 1 level of stack frame."""
    mutter_callsite(1, 'foo %s', 'a string')
    log = self.get_log()
    self.assertLogContainsLine(log, 'foo a string\nCalled from:\n')
    self.assertContainsRe(log, 'test_trace\\.py", line \\d+, in test_mutter_callsite_1\n')
    self.assertEndsWith(log, ' "a string")\n')