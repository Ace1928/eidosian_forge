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
def find_system_log(self):
    """Find the system log file or skip the current test."""
    filename = '/var/log/system.log' if sys.platform == 'darwin' else '/var/log/syslog' if 'linux' in sys.platform else None
    if not filename:
        self.skipTest('Location of system log file unknown!')
    elif not os.path.isfile(filename):
        self.skipTest('System log file not found! (%s)' % filename)
    elif not os.access(filename, os.R_OK):
        self.skipTest('Insufficient permissions to read system log file! (%s)' % filename)
    else:
        return filename