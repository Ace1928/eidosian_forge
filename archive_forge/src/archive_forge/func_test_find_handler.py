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
def test_find_handler(self):
    """Make sure find_handler() works as intended."""
    root, parent, child, grand_child = self.get_logger_tree()
    stream_handler = logging.StreamHandler()
    syslog_handler = logging.handlers.SysLogHandler()
    child.addHandler(stream_handler)
    parent.addHandler(syslog_handler)
    matched_handler, matched_logger = find_handler(grand_child, lambda h: isinstance(h, logging.Handler))
    assert matched_handler is stream_handler
    matched_handler, matched_logger = find_handler(child, lambda h: isinstance(h, logging.handlers.SysLogHandler))
    assert matched_handler is syslog_handler