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
def push_char(self, char: bytes | None, x: int, y: int) -> None:
    """
        Push one character to current position and advance cursor to x/y.
        """
    if char is not None:
        char = self.charset.apply_mapping(char)
        if self.modes.insert:
            self.insert_chars(char=char)
        else:
            self.set_char(char)
    self.set_term_cursor(x, y)