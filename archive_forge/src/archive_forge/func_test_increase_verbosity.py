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
def test_increase_verbosity(self):
    """Make sure increase_verbosity() respects default and custom levels."""
    set_level(logging.INFO)
    assert get_level() == logging.INFO
    increase_verbosity()
    assert get_level() == logging.VERBOSE
    increase_verbosity()
    assert get_level() == logging.DEBUG
    increase_verbosity()
    assert get_level() == logging.SPAM
    increase_verbosity()
    assert get_level() == logging.NOTSET
    increase_verbosity()
    assert get_level() == logging.NOTSET