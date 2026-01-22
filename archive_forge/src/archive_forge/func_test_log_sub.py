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
def test_log_sub(self):
    logging.getLogger('brz.test_log_sub').debug('Whispered')
    self.assertEqual('   DEBUG  Whispered\n', self.get_log())