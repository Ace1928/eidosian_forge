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
def test_format_sockets_error(self):
    try:
        import socket
        sock = socket.socket()
        sock.send(b'This should fail.')
    except OSError:
        msg = _format_exception()
    self.assertNotContainsRe(msg, 'Traceback \\(most recent call last\\):')