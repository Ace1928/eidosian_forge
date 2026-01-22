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
def test_plain_text_output_format(self):
    """Inspect the plain text output of coloredlogs."""
    logger = VerboseLogger(random_string(25))
    stream = StringIO()
    install(level=logging.NOTSET, logger=logger, stream=stream)
    logger.setLevel(logging.INFO)
    logger.debug('No one should see this message.')
    assert len(stream.getvalue().strip()) == 0
    logger.setLevel(logging.NOTSET)
    for method, severity in ((logger.debug, 'DEBUG'), (logger.info, 'INFO'), (logger.verbose, 'VERBOSE'), (logger.warning, 'WARNING'), (logger.error, 'ERROR'), (logger.critical, 'CRITICAL')):
        try:
            logger._cache.clear()
        except AttributeError:
            pass
        text = 'This is a message with severity %r.' % severity.lower()
        method(text)
        output = stream.getvalue()
        lines = output.splitlines()
        last_line = lines[-1]
        assert text in last_line
        assert severity in last_line
        assert PLAIN_TEXT_PATTERN.match(last_line)