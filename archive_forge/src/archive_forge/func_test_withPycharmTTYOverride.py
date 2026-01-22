import sys
from unittest import TestCase, main
from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY
def test_withPycharmTTYOverride(self):
    tty = StreamTTY()
    with pycharm(), replace_by(tty):
        self.assertTrue(is_a_tty(tty))