import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
def repl(self) -> Tuple[Any, ...]:
    """Initialise the repl and jump into the loop. This method also has to
        keep a stack of lines entered for the horrible "undo" feature. It also
        tracks everything that would normally go to stdout in the normal Python
        interpreter so it can quickly write it to stdout on exit after
        curses.endwin(), as well as a history of lines entered for using
        up/down to go back and forth (which has to be separate to the
        evaluation history, which will be truncated when undoing."""
    self.push('from bpython._internal import _help as help\n', False)
    self.iy, self.ix = self.scr.getyx()
    self.more = False
    while not self.do_exit:
        self.f_string = ''
        self.prompt(self.more)
        try:
            inp = self.get_line()
        except KeyboardInterrupt:
            self.statusbar.message('KeyboardInterrupt')
            self.scr.addstr('\n')
            self.scr.touchwin()
            self.scr.refresh()
            continue
        self.scr.redrawwin()
        if self.do_exit:
            return self.exit_value
        self.history.append(inp)
        self.screen_hist[-1] += self.f_string
        self.stdout_hist += inp + '\n'
        stdout_position = len(self.stdout_hist)
        self.more = self.push(inp)
        if not self.more:
            self.prev_block_finished = stdout_position
            self.s = ''
    return self.exit_value