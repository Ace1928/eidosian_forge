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
def test_style_parsing(self):
    """Make sure :func:`~coloredlogs.parse_encoded_styles()` works as intended."""
    encoded_styles = 'debug=green;warning=yellow;error=red;critical=red,bold'
    decoded_styles = parse_encoded_styles(encoded_styles, normalize_key=lambda k: k.upper())
    assert sorted(decoded_styles.keys()) == sorted(['debug', 'warning', 'error', 'critical'])
    assert decoded_styles['debug']['color'] == 'green'
    assert decoded_styles['warning']['color'] == 'yellow'
    assert decoded_styles['error']['color'] == 'red'
    assert decoded_styles['critical']['color'] == 'red'
    assert decoded_styles['critical']['bold'] is True