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
def csi_erase_display(self, mode: Literal[0, 1, 2]) -> None:
    """
        Erase display, modes are:
            0 -> erase from cursor to end of display.
            1 -> erase from start to cursor.
            2 -> erase the whole display.
        """
    if mode == 0:
        self.erase(self.term_cursor, (self.width - 1, self.height - 1))
    if mode == 1:
        self.erase((0, 0), (self.term_cursor[0] - 1, self.term_cursor[1]))
    elif mode == 2:
        self.clear(cursor=self.term_cursor)