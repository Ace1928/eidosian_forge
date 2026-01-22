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
def insert_char_pair_start(self, e):
    """Accepts character which is a part of CHARACTER_PAIR_MAP
        like brackets and quotes, and appends it to the line with
        an appropriate character pair ending. Closing character can only be inserted
        when the next character is either a closing character or a space

        e.x. if you type "(" (lparen) , this will insert "()"
        into the line
        """
    self.add_normal_character(e)
    if self.config.brackets_completion:
        start_of_line = len(self._current_line) == 1
        end_of_line = len(self._current_line) == self._cursor_offset
        can_lookup_next = len(self._current_line) > self._cursor_offset
        next_char = None if not can_lookup_next else self._current_line[self._cursor_offset]
        if start_of_line or end_of_line or (next_char is not None and next_char in '})] '):
            self.add_normal_character(CHARACTER_PAIR_MAP[e], narrow_search=False)
            self._cursor_offset -= 1