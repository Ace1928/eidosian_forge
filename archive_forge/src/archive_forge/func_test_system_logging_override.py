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
def test_system_logging_override(self):
    """Make sure the :class:`coloredlogs.syslog.is_syslog_supported` respects the override."""
    with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'true'):
        assert is_syslog_supported() is True
    with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'false'):
        assert is_syslog_supported() is False