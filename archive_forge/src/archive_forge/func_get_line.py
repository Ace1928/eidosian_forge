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
def get_line(self) -> str:
    """Get a line of text and return it
        This function initialises an empty string and gets the
        curses cursor position on the screen and stores it
        for the echo() function to use later (I think).
        Then it waits for key presses and passes them to p_key(),
        which returns None if Enter is pressed (that means "Return",
        idiot)."""
    self.s = ''
    self.rl_history.reset()
    self.iy, self.ix = self.scr.getyx()
    if not self.paste_mode:
        for _ in range(self.next_indentation()):
            self.p_key('\t')
    self.cpos = 0
    while True:
        key = self.get_key()
        if self.p_key(key) is None:
            if self.config.cli_trim_prompts and self.s.startswith('>>> '):
                self.s = self.s[4:]
            return self.s