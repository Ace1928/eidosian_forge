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
def test_dynamic_stderr_lookup(self):
    """Make sure coloredlogs.install() uses StandardErrorHandler when possible."""
    coloredlogs.install()
    initial_stream = StringIO()
    initial_text = 'Which stream will receive this text?'
    with PatchedAttribute(sys, 'stderr', initial_stream):
        logging.info(initial_text)
    assert initial_text in initial_stream.getvalue()
    subsequent_stream = StringIO()
    subsequent_text = 'And which stream will receive this other text?'
    with PatchedAttribute(sys, 'stderr', subsequent_stream):
        logging.info(subsequent_text)
    assert subsequent_text in subsequent_stream.getvalue()