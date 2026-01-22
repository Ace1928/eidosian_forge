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
def test__open_brz_log_ignores_cache_dir_error(self):
    self.overrideAttr(sys, 'stderr', StringIO())
    self.overrideEnv('BRZ_LOG', None)
    self.overrideEnv('BRZ_HOME', '/no-such-dir')
    self.overrideEnv('XDG_CACHE_HOME', '/no-such-dir')
    self.overrideAttr(trace, '_brz_log_filename')
    logf = trace._open_brz_log()
    if os.path.isdir('/no-such-dir'):
        raise TestSkipped('directory creation succeeded')
    self.assertIs(None, logf)
    self.assertContainsRe(sys.stderr.getvalue(), "failed to open trace file: .* '/no-such-dir'$")