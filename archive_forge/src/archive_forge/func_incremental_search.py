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
def incremental_search(self, reverse=False, include_current=False):
    if self.incr_search_mode == SearchMode.NO_SEARCH:
        self.rl_history.enter(self.current_line)
        self.incr_search_target = ''
    elif self.incr_search_target:
        line = self.rl_history.back(False, search=True, target=self.incr_search_target, include_current=include_current) if reverse else self.rl_history.forward(False, search=True, target=self.incr_search_target, include_current=include_current)
        self._set_current_line(line, reset_rl_history=False, clear_special_mode=False)
        self._set_cursor_offset(len(self.current_line), reset_rl_history=False, clear_special_mode=False)
    if reverse:
        self.incr_search_mode = SearchMode.REVERSE_INCREMENTAL_SEARCH
    else:
        self.incr_search_mode = SearchMode.INCREMENTAL_SEARCH