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
def idle(caller: CLIRepl) -> None:
    """This is called once every iteration through the getkey()
    loop (currently in the Repl class, see the get_line() method).
    The statusbar check needs to go here to take care of timed
    messages and the resize handlers need to be here to make
    sure it happens conveniently."""
    global DO_RESIZE
    if caller.module_gatherer.find_coroutine() or caller.paste_mode:
        caller.scr.nodelay(True)
        key = caller.scr.getch()
        caller.scr.nodelay(False)
        if key != -1:
            curses.ungetch(key)
        else:
            curses.ungetch('\x00')
    caller.statusbar.check()
    caller.check()
    if DO_RESIZE:
        do_resize(caller)