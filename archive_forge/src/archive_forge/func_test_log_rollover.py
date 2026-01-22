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
def test_log_rollover(self):
    temp_log_name = 'test-log'
    trace_file = open(temp_log_name, 'a')
    trace_file.writelines(['test_log_rollover padding\n'] * 200000)
    trace_file.close()
    _rollover_trace_maybe(temp_log_name)
    self.assertFalse(os.access(temp_log_name, os.R_OK))