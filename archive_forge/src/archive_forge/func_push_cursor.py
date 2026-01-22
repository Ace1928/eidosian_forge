from __future__ import annotations
import atexit
import copy
import errno
import fcntl
import os
import pty
import selectors
import signal
import struct
import sys
import termios
import time
import traceback
import typing
import warnings
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from urwid import event_loop, util
from urwid.canvas import Canvas
from urwid.display import AttrSpec, RealTerminal
from urwid.display.escape import ALT_DEC_SPECIAL_CHARS, DEC_SPECIAL_CHARS
from urwid.widget import Sizing, Widget
from .display.common import _BASIC_COLORS, _color_desc_256, _color_desc_true
def push_cursor(self, char: bytes | None=None) -> None:
    """
        Move cursor one character forward wrapping lines as needed.
        If 'char' is given, put the character into the former position.
        """
    x, y = self.term_cursor
    if self.modes.autowrap:
        if x + 1 >= self.width and (not self.is_rotten_cursor):
            self.is_rotten_cursor = True
            self.push_char(char, x, y)
        else:
            x += 1
            if x >= self.width and self.is_rotten_cursor:
                if y >= self.scrollregion_end:
                    self.scroll()
                else:
                    y += 1
                x = 1
                self.set_term_cursor(0, y)
            self.push_char(char, x, y)
            self.is_rotten_cursor = False
    else:
        if x + 1 < self.width:
            x += 1
        self.is_rotten_cursor = False
        self.push_char(char, x, y)