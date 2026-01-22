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
def test_format_sys_exception(self):
    try:
        raise NotImplementedError('time travel')
    except NotImplementedError:
        err = _format_exception()
    self.assertContainsRe(err, '^brz: ERROR: NotImplementedError: time travel')
    self.assertContainsRe(err, 'Breezy has encountered an internal error.')