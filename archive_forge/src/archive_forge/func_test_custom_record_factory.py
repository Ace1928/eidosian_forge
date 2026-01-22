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
def test_custom_record_factory(self):
    """
        Test that custom LogRecord factories are supported.

        This test is a bit convoluted because the logging module suppresses
        exceptions. We monkey patch the method suspected of encountering
        exceptions so that we can tell after it was called whether any
        exceptions occurred (despite the exceptions not propagating).
        """
    if not hasattr(logging, 'getLogRecordFactory'):
        return self.skipTest('this test requires Python >= 3.2')
    exceptions = []
    original_method = ColoredFormatter.format
    original_factory = logging.getLogRecordFactory()

    def custom_factory(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        record.custom_attribute = 3737844653
        return record

    def custom_method(*args, **kw):
        try:
            return original_method(*args, **kw)
        except Exception as e:
            exceptions.append(e)
            raise
    with PatchedAttribute(ColoredFormatter, 'format', custom_method):
        logging.setLogRecordFactory(custom_factory)
        try:
            demonstrate_colored_logging()
        finally:
            logging.setLogRecordFactory(original_factory)
    assert not exceptions