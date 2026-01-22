import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_printer():
    p = Printer(no_print=True)
    text = 'This is a test.'
    good = p.good(text)
    fail = p.fail(text)
    warn = p.warn(text)
    info = p.info(text)
    assert p.text(text) == text
    if SUPPORTS_ANSI and (not NO_UTF8):
        assert good == '\x1b[38;5;2m✔ {}\x1b[0m'.format(text)
        assert fail == '\x1b[38;5;1m✘ {}\x1b[0m'.format(text)
        assert warn == '\x1b[38;5;3m⚠ {}\x1b[0m'.format(text)
        assert info == '\x1b[38;5;4mℹ {}\x1b[0m'.format(text)
    if SUPPORTS_ANSI and NO_UTF8:
        assert good == '\x1b[38;5;2m[+] {}\x1b[0m'.format(text)
        assert fail == '\x1b[38;5;1m[x] {}\x1b[0m'.format(text)
        assert warn == '\x1b[38;5;3m[!] {}\x1b[0m'.format(text)
        assert info == '\x1b[38;5;4m[i] {}\x1b[0m'.format(text)
    if not SUPPORTS_ANSI and (not NO_UTF8):
        assert good == '✔ {}'.format(text)
        assert fail == '✘ {}'.format(text)
        assert warn == '⚠ {}'.format(text)
        assert info == 'ℹ {}'.format(text)
    if not SUPPORTS_ANSI and NO_UTF8:
        assert good == '[+] {}'.format(text)
        assert fail == '[x] {}'.format(text)
        assert warn == '[!] {}'.format(text)
        assert info == '[i] {}'.format(text)