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
def set_char(self, char: bytes, x: int | None=None, y: int | None=None) -> None:
    """
        Set character of either the current cursor position
        or a position given by 'x' and/or 'y' to 'char'.
        """
    if x is None:
        x = self.term_cursor[0]
    if y is None:
        y = self.term_cursor[1]
    x, y = self.constrain_coords(x, y)
    self.term[y][x] = (self.attrspec, self.charset.current, char)