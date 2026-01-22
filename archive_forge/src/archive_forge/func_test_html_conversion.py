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
def test_html_conversion(self):
    """Check the conversion from ANSI escape sequences to HTML."""
    for color_name, ansi_code in ANSI_COLOR_CODES.items():
        ansi_encoded_text = 'plain text followed by %s text' % ansi_wrap(color_name, color=color_name)
        expected_html = format('<code>plain text followed by <span style="color:{css}">{name}</span> text</code>', css=EIGHT_COLOR_PALETTE[ansi_code], name=color_name)
        self.assertEqual(expected_html, convert(ansi_encoded_text))
    expected_html = '<code><span style="color:#FF0">bright yellow</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('bright yellow', color='yellow', bright=True)))
    expected_html = '<code><span style="background-color:#DE382B">red background</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('red background', background='red')))
    expected_html = '<code><span style="background-color:#F00">bright red background</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('bright red background', background='red', bright=True)))
    expected_html = '<code><span style="color:#FFAF00">256 color mode foreground</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('256 color mode foreground', color=214)))
    expected_html = '<code><span style="background-color:#AF0000">256 color mode background</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('256 color mode background', background=124)))
    expected_html = '<code>plain text expected</code>'
    self.assertEqual(expected_html, convert('\x1b[38;5;256mplain text expected\x1b[0m'))
    expected_html = '<code><span style="font-weight:bold">bold text</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('bold text', bold=True)))
    expected_html = '<code><span style="text-decoration:underline">underlined text</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('underlined text', underline=True)))
    expected_html = '<code><span style="text-decoration:line-through">strike-through text</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('strike-through text', strike_through=True)))
    expected_html = '<code><span style="background-color:#FFC706;color:#000">inverse</span></code>'
    self.assertEqual(expected_html, convert(ansi_wrap('inverse', color='yellow', inverse=True)))
    for sample_text in ('www.python.org', 'http://coloredlogs.rtfd.org', 'https://coloredlogs.rtfd.org'):
        sample_url = sample_text if '://' in sample_text else 'http://' + sample_text
        expected_html = '<code><a href="%s" style="color:inherit">%s</a></code>' % (sample_url, sample_text)
        self.assertEqual(expected_html, convert(sample_text))
    reset_short_hand = '\x1b[0m'
    blue_underlined = ansi_style(color='blue', underline=True)
    ansi_encoded_text = '<%shttps://coloredlogs.readthedocs.io%s>' % (blue_underlined, reset_short_hand)
    expected_html = '<code>&lt;<span style="color:#006FB8;text-decoration:underline"><a href="https://coloredlogs.readthedocs.io" style="color:inherit">https://coloredlogs.readthedocs.io</a></span>&gt;</code>'
    self.assertEqual(expected_html, convert(ansi_encoded_text))