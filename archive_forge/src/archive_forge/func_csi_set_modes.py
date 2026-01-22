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
def csi_set_modes(self, modes: Iterable[int], qmark: bool, reset: bool=False) -> None:
    """
        Set (DECSET/ECMA-48) or reset modes (DECRST/ECMA-48) if reset is True.
        """
    flag = not reset
    for mode in modes:
        self.set_mode(mode, flag, qmark, reset)