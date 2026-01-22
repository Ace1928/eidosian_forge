from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def convert_to_tenths(s):
    if s is None:
        return None
    return int((s + 0.05) * 10)