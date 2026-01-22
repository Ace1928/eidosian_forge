import contextlib
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
from humanfriendly.compat import StringIO
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_CSI, ansi_style, ansi_wrap
from humanfriendly.testing import PatchedAttribute, PatchedItem, TestCase, retry
from humanfriendly.text import format, random_string
import coloredlogs
import coloredlogs.cli
from coloredlogs import (
from coloredlogs.demo import demonstrate_colored_logging
from coloredlogs.syslog import SystemLogging, is_syslog_supported, match_syslog_handler
from coloredlogs.converter import (
from capturer import CaptureOutput
from verboselogs import VerboseLogger
def test_find_hostname(self):
    """Make sure :func:`~find_hostname()` works correctly."""
    assert find_hostname()
    fd, temporary_file = tempfile.mkstemp()
    try:
        with open(temporary_file, 'w') as handle:
            handle.write('first line\n')
            handle.write('second line\n')
        CHROOT_FILES.insert(0, temporary_file)
        assert find_hostname() == 'first line'
    finally:
        CHROOT_FILES.pop(0)
        os.unlink(temporary_file)
    try:
        CHROOT_FILES.insert(0, temporary_file)
        assert find_hostname()
    finally:
        CHROOT_FILES.pop(0)