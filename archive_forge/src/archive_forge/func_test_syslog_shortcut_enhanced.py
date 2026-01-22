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
def test_syslog_shortcut_enhanced(self):
    """Make sure that ``coloredlogs.install(syslog='warning')`` works."""
    system_log_file = self.find_system_log()
    the_expected_message = random_string(50)
    not_an_expected_message = random_string(50)
    with cleanup_handlers():
        coloredlogs.install(syslog='error')
        logging.warning('%s', not_an_expected_message)
        logging.error('%s', the_expected_message)
    retry(lambda: check_contents(system_log_file, the_expected_message, True))
    retry(lambda: check_contents(system_log_file, not_an_expected_message, False))