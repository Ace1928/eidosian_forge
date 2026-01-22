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
def test_format_mem_dump(self):
    self.requireFeature(features.meliae)
    debug.debug_flags.add('mem_dump')
    try:
        raise MemoryError()
    except MemoryError:
        msg = _format_exception()
    self.assertStartsWith(msg, 'brz: out of memory\nMemory dumped to ')