import sys
from unittest import TestCase, main
from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY
def test_withPycharmNonTTYOverride(self):
    non_tty = StreamNonTTY()
    with pycharm(), replace_by(non_tty):
        self.assertFalse(is_a_tty(non_tty))