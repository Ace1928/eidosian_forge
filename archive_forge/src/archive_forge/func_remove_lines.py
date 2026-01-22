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
def remove_lines(self, row: int | None=None, lines: int=1) -> None:
    """
        Remove 'lines' number of lines at the specified row, pulling all
        subsequent lines to the top. If no 'row' is specified, the current row
        is used.
        """
    if row is None:
        row = self.term_cursor[1]
    else:
        row = self.scrollregion_start
    if lines == 0:
        lines = 1
    while lines > 0:
        self.term.pop(row)
        self.term.insert(self.scrollregion_end, self.empty_line())
        lines -= 1