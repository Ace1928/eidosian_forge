import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
def on_tab(self, back=False):
    """Do something on tab key
        taken from bpython.cli

        Does one of the following:
        1) add space to move up to the next %4==0 column
        2) complete the current word with characters common to all completions
        3) select the first or last match
        4) select the next or previous match if already have a match
        """

    def only_whitespace_left_of_cursor():
        """returns true if all characters before cursor are whitespace"""
        return not self.current_line[:self.cursor_offset].strip()
    logger.debug('self.matches_iter.matches: %r', self.matches_iter.matches)
    if only_whitespace_left_of_cursor():
        front_ws = len(self.current_line[:self.cursor_offset]) - len(self.current_line[:self.cursor_offset].lstrip())
        to_add = 4 - front_ws % self.config.tab_length
        for unused in range(to_add):
            self.add_normal_character(' ')
        return
    if self.config.brackets_completion:
        on_closing_char, _ = cursor_on_closing_char_pair(self._cursor_offset, self._current_line)
        if on_closing_char:
            self._cursor_offset += 1
    if len(self.matches_iter.matches) == 0:
        self.list_win_visible = self.complete(tab=True)
    if self.matches_iter.is_cseq():
        cursor_and_line = self.matches_iter.substitute_cseq()
        self._cursor_offset, self._current_line = cursor_and_line
        if not self.matches_iter.matches:
            self.list_win_visible = self.complete()
    elif self.matches_iter.matches:
        self.current_match = back and self.matches_iter.previous() or next(self.matches_iter)
        cursor_and_line = self.matches_iter.cur_line()
        self._cursor_offset, self._current_line = cursor_and_line
        self.list_win_visible = True
    if self.config.brackets_completion:
        if self.is_completion_callable(self._current_line):
            self._current_line = self.append_closing_character(self._current_line)