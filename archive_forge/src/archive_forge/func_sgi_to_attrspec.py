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
def sgi_to_attrspec(self, attrs: Sequence[int], fg: int, bg: int, attributes: set[str], prev_colors: int) -> AttrSpec | None:
    """
        Parse SGI sequence and return an AttrSpec representing the sequence
        including all earlier sequences specified as 'fg', 'bg' and
        'attributes'.
        """
    idx = 0
    colors = prev_colors
    while idx < len(attrs):
        attr = attrs[idx]
        if 30 <= attr <= 37:
            fg = attr - 30
            colors = max(16, colors)
        elif 40 <= attr <= 47:
            bg = attr - 40
            colors = max(16, colors)
        elif attr in {38, 48}:
            if idx + 2 < len(attrs) and attrs[idx + 1] == 5:
                color = attrs[idx + 2]
                colors = max(256, colors)
                if attr == 38:
                    fg = color
                else:
                    bg = color
                idx += 2
            elif idx + 4 < len(attrs) and attrs[idx + 1] == 2:
                color = (attrs[idx + 2] << 16) + (attrs[idx + 3] << 8) + attrs[idx + 4]
                colors = 2 ** 24
                if attr == 38:
                    fg = color
                else:
                    bg = color
                idx += 4
        elif attr == 39:
            fg = None
        elif attr == 49:
            bg = None
        elif attr == 10:
            self.charset.reset_sgr_ibmpc()
            self.modes.display_ctrl = False
        elif attr in {11, 12}:
            self.charset.set_sgr_ibmpc()
            self.modes.display_ctrl = True
        elif attr == 1:
            attributes.add('bold')
        elif attr == 4:
            attributes.add('underline')
        elif attr == 5:
            attributes.add('blink')
        elif attr == 7:
            attributes.add('standout')
        elif attr == 24:
            attributes.discard('underline')
        elif attr == 25:
            attributes.discard('blink')
        elif attr == 27:
            attributes.discard('standout')
        elif attr == 0:
            fg = bg = None
            attributes.clear()
        idx += 1
    if 'bold' in attributes and colors == 16 and (fg is not None) and (fg < 8):
        fg += 8

    def _defaulter(color: int | None, colors: int) -> str:
        if color is None:
            return 'default'
        if color > 255 or colors == 2 ** 24:
            return _color_desc_true(color)
        if color > 15 or colors == 256:
            return _color_desc_256(color)
        return _BASIC_COLORS[color]
    decoded_fg = _defaulter(fg, colors)
    decoded_bg = _defaulter(bg, colors)
    if attributes:
        decoded_fg = ','.join((decoded_fg, *list(attributes)))
    if decoded_fg == decoded_bg == 'default':
        return None
    if colors:
        return AttrSpec(decoded_fg, decoded_bg, colors=colors)
    return AttrSpec(decoded_fg, decoded_bg)